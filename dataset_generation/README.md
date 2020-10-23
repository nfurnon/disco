### Generate the DISCO dataset
The DISCO dataset is described in our submitted paper: [DNN-based mask estimation for distributed speech enhancement in spatially unconstrained microphone arrays](). It simulates three kinds of everyday life spatial configurations:
 * A random room where microphones and sources can be placed anywhere in a room;
 * A living room where most of the microphones are placed close to the walls;
 * A meeting room where two sources are around a table and four recording devices on the table.

To generate the dataset:
```bash
bash generate_disco.sh
```

### Generate the Meetit dataset
The Meetit (MEETing with Interferent Talkers) dataset is described in our other submitted paper: [Distributed speech separation in spatially unconstrained microphone arrays](). It simulates typical meeting configurations, where several sources are talking around a table, with their recording devices placed on the same table.
```bash
bash generate_meetit.sh
```
