import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from code_utils.db_utils import stack_talkers
from code_utils.sigproc_utils import vad_oracle_batch, noise_from_signal
from matplotlib.patches import Rectangle, Circle
from pyroomacoustics import circular_2D_array


class RandomRoomSetup:
    """
    Class of a room, including its geometry, the sources and sensors positions.
    The class returns only the nodes centers, in order to build circular arrays.
    """

    def __init__(self, l_range, w_range, h_range, beta_range,
                 n_sensors_per_node, d_mw, d_mn, d_nn, z_range_m,  # Sensors positions
                 d_rnd_mics,
                 n_sources, d_ss, d_sn, d_sw, z_range_s,  # Sources parameters
                 **kwargs):

        # Unchanged attributes once the class instantiated
        self.sensors_per_node = n_sensors_per_node
        self.n_nodes = len(n_sensors_per_node)
        self.d_mn = d_mn
        self.d_mw, self.d_mn = d_mw, d_mn
        self.d_nw = d_mw + d_mn
        self.d_rnd_mics = d_rnd_mics
        self.d_nn = d_nn
        self.n_sources = n_sources
        self.d_ss = d_ss  # Source-to-source distance
        self.d_sn = d_sn  # Source-to-node distance
        self.d_sw = d_sw  # Source-to-wall distance
        self.z_range_m, self.z_range_s = z_range_m, z_range_s
        self.l_range, self.w_range, self.h_range, self.beta_range = l_range, w_range, h_range, beta_range
        # Attributes susceptible to change
        self.length, self.width, self.height, self.alpha, self.beta = None, None, None, None, None
        self.nodes_centers, self.source_positions, self.microphones_positions = None, None, None

    def create_room_setup(self):
        # Nodes centres
        find_nodes_centers, find_source_positions = 1, 1
        draw_new_config = [find_nodes_centers, find_source_positions]
        while np.any(draw_new_config):
            self.length, self.width, self.height, self.alpha, self.beta = self.set_room_dimensions(self.l_range,
                                                                                                   self.w_range,
                                                                                                   self.h_range,
                                                                                                   self.beta_range)
            # Sensor nodes
            nodes_centers, n_trials_nodes = self.get_nodes_centers()
            if n_trials_nodes < 100:
                find_nodes_centers = 0
                self.nodes_centers = nodes_centers
                # Sources positions
                source_positions, n_trials_sources = self.get_source_positions()
                if n_trials_sources < 100:
                    self.source_positions = source_positions
                    find_source_positions = 0
            draw_new_config = [find_nodes_centers, find_source_positions]

        # Microphone positions
        self.microphones_positions = self._add_circular_microphones()

    @staticmethod
    def set_room_dimensions(l_range, w_range, h_range, beta_range):
        # Geometric properties
        length = l_range[0] + (l_range[1] - l_range[0]) * np.random.rand()
        width = w_range[0] + (w_range[1] - w_range[0]) * np.random.rand()
        height = h_range[0] + (h_range[1] - h_range[0]) * np.random.rand()
        vol = length * width * height
        sur = 2 * (length * width) + 2 * (length * height) + 2 * (width * height)

        # Acoustic properties
        beta = beta_range[0] + (beta_range[1] - beta_range[0]) * np.random.rand()
        alpha = 1 - np.exp((1.7e-5 * beta - 0.1611) * vol / (beta * sur))

        return length, width, height, alpha, beta

    def get_nodes_centers(self):
        """
        Returns the center of the nodes of the array. They are randomly put in the room, under the constraints:
            - at least d_mw + d_mn from thz_se  walls (so that all microphones are at least d_mw from walls)
            - at least d_nodes from each other (in the z=z_m plane)
        :return:
        A n_nodes x 3 array, gathering for all nodes the (x, y, z) coordinates of their centre.
        """
        nodes_centers = np.zeros((self.n_nodes, 3))
        # Initialize
        x_0 = self.d_nw + (self.length - 2 * self.d_nw) * np.random.rand()
        y_0 = self.d_nw + (self.width - 2 * self.d_nw) * np.random.rand()
        z_0 = self.z_range_m[0] + (self.z_range_m[1] - self.z_range_m[0]) * np.random.rand()

        nodes_centers[0, :] = x_0, y_0, z_0
        n_trials = 0  # Counter to avoid searching endlessly
        # Loop over all centers left
        for i_n in range(1, self.n_nodes):
            x_ = self.d_nw + (self.length - 2 * self.d_nw) * np.random.rand()
            y_ = self.d_nw + (self.width - 2 * self.d_nw) * np.random.rand()
            z_ = self.z_range_m[0] + (self.z_range_m[1] - self.z_range_m[0]) * np.random.rand()
            d2_between_nodes = np.sum((nodes_centers[:i_n, :2] - np.array([x_, y_])) ** 2, axis=1)  # Take d²
            while np.any(d2_between_nodes < self.d_nn ** 2) and n_trials < 100:  # Check not too close to other nodes
                x_ = self.d_nw + (self.length - 2 * self.d_nw) * np.random.rand()
                y_ = self.d_nw + (self.width - 2 * self.d_nw) * np.random.rand()
                d2_between_nodes = np.sum((nodes_centers[:i_n, :2] - np.array([x_, y_])) ** 2, axis=1)
                n_trials += 1
            if n_trials < 100:
                nodes_centers[i_n, :] = x_, y_, z_
                n_trials = 0
            else:
                return nodes_centers, n_trials

        return nodes_centers, n_trials

    def get_random_mics_positions(self):
        """
        Return the (x, y, z) coordinates of two microphones randomly placed in the room, at least distant of self.d_wal
        from the walls and self.d_rnd_mic from each other.
        This is done in a separate function because we will use these microphones only to create diffuse noise
        (by convolving the noise with the sum of the reverberation tails).
        So the microphones may be close to the actually used microphones.
        :return:
        """
        m1_x = self.d_nw + (self.length - 2 * self.d_nw) * np.random.rand()
        m1_y = self.d_nw + (self.width - 2 * self.d_nw) * np.random.rand()
        z_ = self.z_range_m[0] + (self.z_range_m[1] - self.z_range_m[0]) * np.random.rand()

        m2_x = self.d_nw + (self.length - 2 * self.d_nw) * np.random.rand()
        m2_y = self.d_nw + (self.width - 2 * self.d_nw) * np.random.rand()

        while np.sqrt((m1_x - m2_x) ** 2 + (m1_y - m2_y) ** 2) < self.d_rnd_mics:
            m2_x = self.d_nw + (self.length - 2 * self.d_nw) * np.random.rand()
            m2_y = self.d_nw + (self.width - 2 * self.d_nw) * np.random.rand()

        return [m1_x, m1_y, z_], [m2_x, m2_y, z_]

    def get_source_positions(self):
        """
        Get positions of the sources. They are randomly placed in the room, with the following constraints:
            - at least d_sw from the walls (so that all microphones are at least d_wal from walls)
            - at least d_sn from all other sources
            - at least d_ss from each other (in the z=z_s plane)

        :return:
            - Sources positions (x, y, z)
            - a counter: if equal to 100, no configuration was found and new input arguments should be given
        """
        sources_positions = np.zeros((self.n_sources, 3))
        # Initialize
        x_0 = self.d_sw + (self.length - 2 * self.d_sw) * np.random.rand()
        y_0 = self.d_sw + (self.width - 2 * self.d_sw) * np.random.rand()
        z_0 = self.z_range_s[0] + (self.z_range_s[1] - self.z_range_s[0]) * np.random.rand()
        d2_to_nodes = np.sum((self.nodes_centers[:, :2] - np.array([x_0, y_0])) ** 2, axis=1)
        n_trials = 0
        while np.any(d2_to_nodes < self.d_sn ** 2) and n_trials < 100:
            x_0 = self.d_sw + (self.length - 2 * self.d_sw) * np.random.rand()
            y_0 = self.d_sw + (self.width - 2 * self.d_sw) * np.random.rand()
            d2_to_nodes = np.sum((self.nodes_centers[:, :2] - np.array([x_0, y_0])) ** 2, axis=1)
            n_trials += 1
        if n_trials < 100:
            sources_positions[0, :] = x_0, y_0, z_0
            n_trials = 0  # Counter to avoid searching endlessly
        else:
            return sources_positions, n_trials
        # Loop over all sources left
        for i_s in range(1, self.n_sources):
            x_ = self.d_sw + (self.length - 2 * self.d_sw) * np.random.rand()
            y_ = self.d_sw + (self.width - 2 * self.d_sw) * np.random.rand()
            z_ = self.z_range_s[0] + (self.z_range_s[1] - self.z_range_s[0]) * np.random.rand()
            d2_between_sources = np.sum((sources_positions[:i_s, :2] - np.array([x_, y_])) ** 2, axis=1)  # Take d²
            d2_to_nodes = np.sum((self.nodes_centers[:, :2] - np.array([x_, y_])) ** 2, axis=1)
            while (np.any(d2_between_sources < self.d_ss ** 2) or np.any(d2_to_nodes < self.d_sn ** 2)) \
                    and n_trials < 100:
                x_ = self.d_sw + (self.length - 2 * self.d_sw) * np.random.rand()
                y_ = self.d_sw + (self.width - 2 * self.d_sw) * np.random.rand()
                d2_between_sources = np.sum((sources_positions[:i_s, :2] - np.array([x_, y_])) ** 2, axis=1)
                d2_to_nodes = np.sum((self.nodes_centers[:, :2] - np.array([x_, y_])) ** 2, axis=1)
                n_trials += 1
            if n_trials < 100:
                sources_positions[i_s, :] = x_, y_, z_
                n_trials = 0
            else:
                return sources_positions, n_trials
        return sources_positions, n_trials

    def _add_circular_microphones(self):
        """
        Add the microphones to the room class, by creating circular 2D arrays around the nodes centres.
        The height of microphones is constant at self.z_m.
        :return:
        """
        n_mics = np.sum(self.sensors_per_node)  # Total number of microphones
        node_mics = np.zeros((3, n_mics))  # Array of mics (has to be (x, y [, z]) x n_mics for pra.mic_array)
        mics_created = 0
        for i_n in range(self.n_nodes):
            local_mics = np.zeros((3, self.sensors_per_node[i_n]))  # See pra.MicrophoneArray: n_mic is second dime
            local_mics[:2, :] = circular_2D_array(center=self.nodes_centers[i_n][:2],
                                                  M=self.sensors_per_node[i_n],
                                                  phi0=np.pi / 2 * np.random.rand(),
                                                  radius=self.d_mn)
            local_mics[-1, :] = self.nodes_centers[i_n][-1]
            node_mics[:, mics_created:mics_created + self.sensors_per_node[i_n]] = local_mics
            mics_created += self.sensors_per_node[i_n]

        return node_mics

    def plot_room(self):
        plt.figure()
        plt.gca().add_patch(Rectangle((0, 0), self.length, self.width, fill=False, linewidth=3))
        plt.plot(self.nodes_centers[:, 0], self.nodes_centers[:, 1], 'x')
        plt.plot(self.source_positions[:, 0], self.source_positions[:, 1], 'o')
        plt.gca().axis('equal')
        # Text
        for i_n in range(self.n_nodes):
            plt.text(1.05 * self.nodes_centers[i_n, 0], 1.05 * self.nodes_centers[i_n, 1],
                     'Node ' + str(i_n + 1), fontsize=4)
        for i_s in range(self.n_sources):
            plt.text(1.05 * self.source_positions[i_s, 0], 1.05 * self.source_positions[i_s, 1],
                     'Source ' + str(i_s + 1), fontsize=4)
        plt.gca().set(xlim=(-1, self.length + 1), ylim=(-1, self.width + 1))
        plt.show()


class MeetitSetup(RandomRoomSetup):
    """
    Class of a meeting room setup, where a table is in the room, 4 nodes placed on the table and two sources around
    the table. A third noise source is located somewhere close to the walls.
    First noise source is interferent speaker, the second is another environmental noise (running water, copy-machine,
     ...)
    """

    def __init__(self, r_range, d_nt_range, d_st_range, **kwargs):
        """
        :param r_range:     Table radius in meters
        :param d_nt_range   Distance range between nodes and table edge
        :param d_st_range   Distance range between sources and table edge.
                            NB: it should not be too high, as this would restrict the possible locations of the table
        :param phi_ss:      Either a scalar or an array/list.
                                - If scalar: maximum angle between the two sources around the table, IN RAD
                                - If array: limited choices of accepted angles between the two sources, IN RAD
        """
        RandomRoomSetup.__init__(self, **kwargs)
        self.r_range = r_range
        self.d_nt_range, self.d_st_range = d_nt_range, d_st_range
        self.d_nt, self.d_st, self.phi_t, self.d_max = None, None, None, None
        self.table_center, self.table_radius = None, None

    def get_table_position(self):
        """
        Place the table center and chose its diameter.
        :return:
        """
        # Fix dimension of table
        r = self.r_range[0] + (self.r_range[1] - self.r_range[0]) * np.random.rand()
        # Fix distances of nodes and sources to table
        self.d_max = np.minimum(self.d_nt_range[1], r - self.d_mn)  # If radius is small, avoid going beyond the enter
        self.d_nt = self.d_max / 2
        self.d_st = self.d_st_range[0] + (self.d_max - self.d_st_range[0]) * np.random.rand()
        dt_min = self.d_sw + self.d_st + r  # Minimal distance to walls
        x_t = dt_min + (self.length - 2 * dt_min) * np.random.rand()  # Table center x
        y_t = dt_min + (self.width - 2 * dt_min) * np.random.rand()  # Table center y
        z_t = self.z_range_m[0] + (self.z_range_m[1] - self.z_range_m[0]) * np.random.rand()  # Take nodes height

        self.table_center = (x_t, y_t, z_t)  # Table height is the same as that of mics
        self.table_radius = r
        return self.table_center, self.table_radius, self.d_max

    def get_nodes_centers(self):
        """
        Place the nodes centers on the table
        :return:
        """
        # Preallocate
        nodes_centers = np.zeros((self.n_nodes, 3))
        # Parameters
        table_center, table_radius, _ = self.get_table_position()
        self.phi_t = 2 * np.pi / self.n_nodes * np.random.rand()  # Central symmetry --> Redundant angles
        # Position
        nodes_centers[:, :2] = circular_2D_array(table_center[:2], self.n_nodes, self.phi_t, table_radius - self.d_nt).T
        # Random shift (to avoid perfect circle)
        angles_proj = self.get_nodes_angles()[-1]
        coords_to_add = -self.d_nt + (self.d_max - self.d_nt_range[0]) * np.random.rand(self.n_sources, 1)
        nodes_centers[:, :2] += coords_to_add * angles_proj
        # Set height
        nodes_centers[:, -1] = table_center[-1]  # Nodes on the table

        return nodes_centers, 0

    def get_nodes_angles(self):
        """Return the angles of each node relative to the table center.
        Returns:
            angles (np.ndarray): angles of each node in rad (n_nodes x 1)
            proj (np.ndarray): projections of the vector on the x, y axis (n_nodes x 2)
        """
        angles = self.phi_t + np.linspace(0, 2 * (self.n_sources - 1) * np.pi / self.n_sources, self.n_sources)
        proj = np.array([np.cos(angles), np.sin(angles)]).T
        return angles, proj

    def get_source_positions(self):
        """
        Place sources around the table. self.node_centers should aready be not None.
        Returns:
            - sources_positions (np.ndarray):   sources positions (n_sources x (x, y, z))
            - counter (0): As the scenario is constrained, the sources positions are always valid, the counter is 0
        """
        sources_positions = np.zeros((self.n_nodes, 3))
        sources_positions[:, :2] = circular_2D_array(self.table_center[:2],
                                                     self.n_nodes,
                                                     self.phi_t,
                                                     self.table_radius + self.d_st).T
        for i_s in range(self.n_sources):
            sources_positions[i_s, -1] = self.z_range_s[0] + (self.z_range_s[1] - self.z_range_s[0]) * np.random.rand()
        # Check that the sources are in the room
        n_trials = 0
        if (np.any(sources_positions[:, :2] <= self.d_sw)\
            or np.any(sources_positions[:, 0] >= self.length - self.d_sw)\
            or np.any(sources_positions[:, 1] >= self.width - self.d_sw)):
            n_trials = 100

        return np.array(sources_positions), n_trials

    def plot_room(self):
        plt.figure()
        plt.gca().add_patch(Rectangle((0, 0), self.length, self.width, fill=False, linewidth=3))
        plt.plot(self.nodes_centers[:, 0], self.nodes_centers[:, 1], 'x')
        plt.plot(self.source_positions[:, 0], self.source_positions[:, 1], 'o')
        plt.gca().add_patch(Circle(self.table_center[:2], self.table_radius, fc='none', ec='black', linewidth=1))
        plt.gca().axis('equal')
        # Text
        for i_n in range(self.n_nodes):
            plt.text(1.05 * self.nodes_centers[i_n, 0], 1.05 * self.nodes_centers[i_n, 1],
                     'Node ' + str(i_n + 1), fontsize=4)
        for i_s in range(self.n_sources):
            plt.text(1.05 * self.source_positions[i_s, 0], 1.05 * self.source_positions[i_s, 1],
                     'Source ' + str(i_s + 1), fontsize=4)
        plt.gca().set(xlim=(-1, self.length + 1), ylim=(-1, self.width + 1))
        plt.show()


class SignalSetup:
    """
    Class of setup for what deals with the signals: SNR, corpus they are taken from, duration.
    It is based on a list-logic, that is to say that the WAV files are picked among the limited possibilities of a list.
    This is because the database generation is expected to be run on parallel processes, so we want to avoid taking two
    times the same WAV file.
    """

    def __init__(self, speakers_lists, duration_range, var_tar, snr_dry_range, snr_cnv_range,
                 min_delta_snr):
        self.speakers_list = speakers_lists
        self.duration_range = duration_range  # min_dur, max_dur of signals (signals > min are padded to max)
        self.target_duration = None  # Duration of target signal -- determined in get_target_segment
        self.speakers_ids, self.speakers_files = [], []
        self.var_tar = var_tar  # Normalized variance of all target signals
        self.snr_dry_range = snr_dry_range  # SNR range of dry signals (at loudspeakers)
        self.snr_cnv_range = snr_cnv_range  # SNR range of convolved signals (at microphones)
        self.min_delta_snr = min_delta_snr  # Maximum difference of SNRs between nodes
        self.source_snr = np.zeros(np.shape(snr_dry_range)[0])

    def get_target_segment(self, target_file):
        """
        Return source signals (one noise, one target)
        :param target_file:     name of an audio file
        :return:                If target_file is long enough, the reshaped signal; if too short, None
        """
        min_duration, max_duration = self.duration_range[0], self.duration_range[1]
        signal, fs = sf.read(target_file)
        signal = signal[:int(max_duration * fs)]
        signal -= np.mean(signal)       # Some librispeech files are not zero-meaned
        sig_duration = len(signal) / fs

        if sig_duration < min_duration:
            signal = None
            vad_signal = None
        else:
            # VAD
            vad_signal = vad_oracle_batch(signal, thr=0.001)
            # Normalize the segment
            signal *= np.sqrt(self.var_tar / np.var(signal[vad_signal == 1]))
            # Update VAD (because of energy, no linear process and VAD is different)
            vad_signal = vad_oracle_batch(signal, thr=0.001)

        self.target_duration = sig_duration
        # Get the speaker ID to not take it again for other sources
        speaker_id = int(target_file.split('/')[-3])
        self.speakers_ids.append(speaker_id)
        self.speakers_files.append(target_file)

        return signal, vad_signal, fs

    def get_noise_segment(self, duration):
        assert (duration > 0), "Duration should be strictly positive"
        sig_duration, n_trials = 0, 0
        max_trials = 100
        speaker_is_known = True
        while sig_duration < duration and n_trials < max_trials or speaker_is_known:
            speaker_file = np.random.choice(self.speakers_list)
            sig_duration = sf.info(speaker_file).duration
            speaker_id = speaker_file.split('/')[-3]
            speaker_is_known = (speaker_id in self.speakers_ids)
            n_trials += 1
        if n_trials == max_trials:
            raise ValueError("Failed to find a file lasting more that {} s. "
                             "Please choose a shorter duration".format(duration))
        else:
            sig, fs = sf.read(speaker_file)
            y = sig[:np.int(duration * fs)]              # signal at the end.
            y -= np.mean(y)
            # VAD
            vad_signal = vad_oracle_batch(y, thr=0.001)
            # Normalize the segment
            y *= np.sqrt(self.var_tar / np.var(y[vad_signal == 1]))
            # Update VAD (because of energy, no linear process and VAD is different)
            vad_signal = vad_oracle_batch(y, thr=0.001)
            self.speakers_ids.append(speaker_id)
            self.speakers_files.append(speaker_file)

        return y, vad_signal

