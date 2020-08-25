import os
import torch
from torch.utils.data import Dataset
import numpy as np

TRAIN_DUR = 11  # seconds
FS = 16000  # Hz
STFT_MIN = 1e-6
STFT_MAX = 1e3


# %%
class RandomDataset(Dataset):
    """
    Subclass of torch.utils.data.Dataset. It can be passed to a torch.utils.data.DataLoader
    which can load multiple samples parallelly using torch.multiprocessing workers.
    To feed LSTM, what this class returns should be of shape (seq_len, batch, input_size)
    unless batch_first is set to True, in which case shape should be (batch, seq_len, input_size)
    """

    def __init__(self, input_shape, output_shape, length=1000):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.length = length

    def __len__(self):
        """
        Total number of samples in the dataset.
        """
        return self.length

    def __getitem__(self, index):
        x = np.random.random(self.input_shape).astype('float32')
        y = np.random.random(self.output_shape).astype('float32')
        return torch.from_numpy(x), torch.from_numpy(y)


# %%
class DiscoDataset(Dataset):
    def __init__(self, lists_to_load, stack_axis=0, z_nodes=None,
                 fft_len=512, fft_hop=256, win_len=21, win_hop=8):
        """
        Return windows of `win_len` frames long, by hop size of `win_hop`.
        The segments are loaded from the signals indicated in `lists_to_load` and the windows in the segment are
        determined by `item` in `__getitem__`.
        """
        super(DiscoDataset, self).__init__()
        # STFT parameters
        self.n_fft = fft_len
        self.n_hop = fft_hop
        self.n_freq = int(self.n_fft // 2) + 1
        # Subwindows parameters
        self.win_len = win_len
        self.win_hop = win_hop
        # Data to load
        self.n_nodes = 4
        self.segs_to_load = lists_to_load
        self.n_ch = np.shape(lists_to_load)[0] - 1  # All segs_to_load are input to stack except mask
        assert stack_axis in [0, 1, 2], "`stack_axis` should be 0 (SC), 1 (MC stacked on freq axis) or 2 (MC stacked " \
                                        "over the channels)"
        self.stack_axis = stack_axis
        if z_nodes is None:
            self.z_nodes = np.minimum(self.stack_axis, 1) * (self.n_nodes - 1)
        else:
            self.z_nodes = z_nodes

        self.data, self.first_seq_frame, self.win_per_seg, self.n_frames = self.load_data()
        self.n_cum = np.cumsum([0] + list(self.win_per_seg))  # Cumulated number of possible frames

    def load_data(self):
        first_seq_frame = int(np.ceil(FS / self.n_hop))
        n_frames_max = (TRAIN_DUR * FS - self.n_fft) // self.n_hop + 3 - first_seq_frame  # +3 because of center=True
        win_per_seg = np.zeros(len(self.segs_to_load[0]), 'int')  # For class length
        n_frames = np.zeros(len(self.segs_to_load[0]), 'int')  # Do not load the last zero-frames
        data = np.zeros((np.shape(self.segs_to_load)[0], np.shape(self.segs_to_load)[1], self.n_freq, n_frames_max),
                        'float32')
        for i_rir in range(len(self.segs_to_load[0])):
            for i_sig in range(np.shape(self.segs_to_load)[0]):
                tmp_load = abs(np.load(self.segs_to_load[i_sig][i_rir]))
                tmp_load = tmp_load[:, first_seq_frame:]  # Do not load first second of silence
                data[i_sig, i_rir, :, :tmp_load.shape[1]] = tmp_load
                if i_sig == 0:
                    n_frames[i_rir] = tmp_load.shape[1]
                    win_per_seg[i_rir] = int((tmp_load.shape[1] - self.win_len) // self.win_hop + 1)

        return data, first_seq_frame, win_per_seg, n_frames

    def __len__(self):
        return int(sum(self.win_per_seg))

    def __getitem__(self, item):
        """
        Load the input and label corresponding to `item`.
        A window is determined by the index of the segment it is picked from and by the index of its first frame
        (where does the window start in this segment ?). They both depend on `item` only.
        """
        segment_index, frame_index = self.get_item_indices(item)
        local_node = np.random.randint(self.n_nodes)  # Node where working = which receives the compressed signals
        mixture_normed, mask = self.get_subwindow(local_node, segment_index, frame_index)

        return torch.from_numpy(np.swapaxes(mixture_normed, -2, -1)), \
               torch.from_numpy(mask.T)

    def get_item_indices(self, item):
        """
        Return index of the file corresponding to `item`, and the sample index corresponding to the first frame of the
        window.
        Args:
            item (int): Item of the sample to load
        Returns:
            k (int): Segment index the window is picked from
            m (int): Frame index in the segment of the first window frame.
        """
        k = np.where(self.n_cum > item)[0][0] - 1  # Index of the STFT-sequence corresponding to item
        m = int(item - self.n_cum[k]) * self.win_hop + np.random.randint(self.win_hop)  # Frame index
        m = np.minimum(m, self.n_frames[k] - self.win_len)  # In some cases, m > len - win_len
        return k, m

    def get_subwindow(self, local_node, k, m):
        """
        Load the frames of the k-th segment comprised between m and m+self.win_len
        Args:
            local_node (int):  Index of the node we are working on (which receives the compressed signals)
            k (int):           Index of the segment to load (in self.segs_to_load)
            m (int):           Index of the first frame to load
        Returns:
           mixt: input (2D if self.stack_axis = 1; 3D if self.stack_axis=2)
           mask: label (masks, 2D).
        """
        # Load local reference (not compressed)
        mixt = [self.data[local_node, k, :, m:m + self.win_len]]
        # Load compressed signals of other nodes
        z_chs = np.arange(self.n_nodes)
        if self.z_nodes == 1:
            z_chs = list(z_chs)
            z_chs.pop(local_node)
            z_chs = np.random.permutation(np.array(z_chs))
        else:
            z_chs = np.roll(z_chs, self.n_nodes - 1 - local_node)  # Put local node index at last index
        for i_ch in range(self.z_nodes):
            for i_sig in range(int(np.shape(self.segs_to_load)[0] / self.n_nodes) - 2):  # Only zs/zn or both ?
                mixt.append(self.data[self.n_nodes * (i_sig + 1) + z_chs[i_ch], k, :, m:m + self.win_len])
        mixt = np.squeeze(np.array(mixt))

        if self.stack_axis == 1:
            mixt = np.concatenate((mixt[0, :, :], *mixt[1:, :, :]), axis=0)

        mask = self.get_mask_frames(local_node, k, m)

        return abs(mixt), mask

    def get_mask_frames(self, local_node, k, m):
        """ In `self.data`, get the frames of the label corresponding to `item`.
        Args:
            local_node (int):  Index of the node we are working on (which receives the compressed signals)
            k (int):           Index of the segment to load (in self.segs_to_load)
            m (int):           Index of the first frame to load
        Returns:
            The `win_len` long label array corresponding to the input determined by `item`.
        """
        return self.data[-self.n_nodes + local_node, k, :, m:m + self.win_len]


class DiscoPartialDataset(DiscoDataset):
    """Instead of loading the whole dataset into a variable (so into RAM), load only the compressed signals in the
        variable and pick the other ones (reference signal, label mask) on the fly on disk."""

    def __init__(self, *args, **kwargs):
        super(DiscoPartialDataset, self).__init__(*args, **kwargs)

    def load_data(self):
        """Load only the compressed signals into self.data"""
        first_seq_frame = int(np.ceil(FS / self.n_hop))
        n_frames_max = (TRAIN_DUR * FS - self.n_fft) // self.n_hop + 3 - first_seq_frame  # +3 because of center=True
        win_per_seg = np.zeros(len(self.segs_to_load[0]), 'int')  # For class length
        n_frames = np.zeros(len(self.segs_to_load[0]), 'int')  # Do not load the last zero-frames
        data = np.zeros((np.shape(self.segs_to_load)[0] - 2 * self.n_nodes,  # Do not count first channel and output
                         np.shape(self.segs_to_load)[1],
                         self.n_freq,
                         n_frames_max), 'float32')
        for i_rir in range(len(self.segs_to_load[0])):
            for i_sig in range(np.shape(data)[0]):
                tmp_load = abs(np.load(self.segs_to_load[i_sig + self.n_nodes][i_rir]))  # Skip reference channels
                tmp_load = tmp_load[:, first_seq_frame:]  # Do not load first second of silence
                data[i_sig, i_rir, :, :tmp_load.shape[1]] = tmp_load
                if i_sig == 0:
                    n_frames[i_rir] = tmp_load.shape[1]
                    win_per_seg[i_rir] = int((tmp_load.shape[1] - self.win_len) // self.win_hop + 1)

        return data, first_seq_frame, win_per_seg, n_frames

    def get_subwindow(self, local_node, k, m):
        """Load reference channel and mask from disk memory. Load compressed signals from self.data."""
        # Compensate that first second of silent was not loaded in self.data
        m_ = m + self.first_seq_frame
        # Load local reference (not compressed)
        mixt = [abs(np.load(self.segs_to_load[local_node][k])[:, m_:m_ + self.win_len]).astype('float32')]
        # Load compressed signals of other nodes
        z_chs = np.arange(self.n_nodes)
        if self.z_nodes == 1:
            z_chs = list(z_chs)
            z_chs.pop(local_node)
            z_chs = np.random.permutation(np.array(z_chs))
        else:
            z_chs = np.roll(z_chs, self.n_nodes - 1 - local_node)  # Put local node index at last index
        for i_ch in range(self.z_nodes):
            for i_sig in range(int(np.shape(self.segs_to_load)[0] / self.n_nodes) - 2):  # Only zs/zn or both ?
                mixt.append(self.data[self.n_nodes * i_sig + z_chs[i_ch], k, :, m:m + self.win_len])
        mixt = np.squeeze(np.array(mixt))

        if self.stack_axis == 1:
            mixt = np.concatenate((mixt[0, :, :], *mixt[1:, :, :]), axis=0)

        mask = self.get_mask_frames(local_node, k, m_)

        return abs(mixt), mask

    def get_mask_frames(self, local_node, k, m):
        """ Load from the disk the `win_len` long label array corresponding to the input determined by `item`."""
        return np.load(self.segs_to_load[-self.n_nodes + local_node][k])[:, m:m + self.win_len].astype('float32')

