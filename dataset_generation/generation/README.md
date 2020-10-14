## Generation of the DISCO dataset
Assumed is that the noise and speech files are stored at an accessible place, mentioned lines 396, 397. To gather the noise files, you may [download them from freesound](../pre_generation/README.md).

The dataset is generated in two steps: 
  1. Simulate the rooms and convolve the signals at an SNR of 0dB.
  2. Mix the image signals for a given noise at a random SNR between a given range.

__Example__:
  1. Simulate the rooms  
  ```bash 
  python generate_disco.py --dset train --scenario random -i 1 -n 10 -d tmp
  ```

  2. Mix the image signals
  ```bash 
  python post_generation.py -r 1 10 -s random -n ssn
  ```


__Caveat__:  

 * The RIR IDs should be different for the different datasets train/val/test, because the pseudo-random room sizes depend on the RIR ID.

 * Some target files are too short to be considered. In this case, the target file identifier is increased by 1. Overall, less than half of the speech files are concerned, that is why, when running parallel processes, it is an acceptable margin to start selecting the speech files at an identifier equal to twice the RIR ID. __However__, since we do not know how many speech files were actually too short to be considered, we cannot not a posteriori what speech files have already been used. Therefore, it is not possible, (unless the speech files IDs are stored, which we do not do here), to re-launch the script for a restricted number for RIR IDs while making sure that the speech files that are going to be picked have not already been picked by a previous process.  
Ex: We create RIRs 1, 2, 3. If for a reason or another, I want to recreate RIR 2, I cannot make sure that the speech file that will be picked has not already been used for RIR 3. I can be relatively sure however that it has not been picked for RIR 1.
