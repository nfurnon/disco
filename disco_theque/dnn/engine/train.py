"""Training pipeline"""
import os
import torch
import time
from torch.utils.data import DataLoader
import argparse
import numpy as np
from disco_theque.dnn.utils import get_model_name
from disco_theque.dnn.utils import get_input_lists, train_one_batch, eval_one_batch, load_architecture, load_states
from disco_theque.dnn.data.lists_to_load import load_input_lists
from disco_theque.dnn.data.datasets import DiscoPartialDataset, DiscoDataset
from disco_theque.dnn.engine.callbacks import SaveAndStop


n_fft = 512
val_split = 0.0909


if __name__ == "__main__":
    np.random.seed(26)
    print('start')
    parser = argparse.ArgumentParser(description="Parameters to train CRNN")
    parser.add_argument("--scene",
                        help="Living or meeting room configuration ?",
                        type=str,
                        default="living")
    parser.add_argument("--noise",
                        choices=['ssn', 'it', 'fs', 'noit', 'all'],
                        default='ssn')
    parser.add_argument("--zsigs", "-zs",
                        type=str,
                        nargs='+',
                        default=['zs_hat'])
    parser.add_argument('--weights', '-w',
                        help='Path to pre-trained networks weights',
                        default='None',
                        type=str)
    parser.add_argument('--files_to_load', '-f2l',
                        type=str,
                        help="Folder where all the files to load are written",
                        default="None")
    parser.add_argument('--zfile', '-zf',
                        help='Folder where z are saved',
                        type=str,
                        default='oracle')
    parser.add_argument('--n_files', '-n',
                        help='Number of sequences to use for training',
                        type=int,
                        default=11001)
    parser.add_argument('--n_epochs', '-epo',
                        help='Number of epochs to train',
                        type=int,
                        default=150)
    parser.add_argument('--path_data', '-path',
                        default=None)
    args = parser.parse_args()
    scene = args.scene
    noise = args.noise
    zsigs = args.zsigs
    zfile = args.zfile
    n_files = args.n_files
    n_epochs = args.n_epochs
    state_dicts = None if args.weights == 'None' else args.weights
    files_lists = None if args.files_to_load == 'None' else args.files_to_load
    path_data = None if args.path_data == "None" else args.path_data
    # Parameters
    case = "train"
    # Data
    if path_data is None:
        path_to_dataset = os.path.join('/home', 'nfurnon', 'dataset', 'suma', '')
    else:
        path_to_dataset = path_data
    archi = 'crnn'
    n_nodes = 4
    win_len = 21
    win_hop = 8
    batch_size = 500
    training_ids = np.arange(1, n_files)
    output_frames = 'all'
    save_path = 'models/'
    # Lists
    zsigs = None if zsigs[0] == "None" else zsigs
    n_ch = 1 if zsigs is None else 1 + len(zsigs) * (n_nodes - 1)
    stax = 0 if zsigs is None else 2
    dataset = DiscoDataset if zsigs is None else DiscoPartialDataset
    if files_lists is None:
        lists_to_load = get_input_lists(path_to_dataset, training_ids,
                                        scenes=scene, noise_to_get=noise, z_sigs=zsigs, z_file=zfile)
    else:
        lists_to_load = load_input_lists(files_lists)
        n_files = len(lists_to_load[0])
    train_list = np.array(lists_to_load)[:, :int(np.ceil((1 - val_split) * n_files))]
    val_list = np.array(lists_to_load)[:, int(np.ceil((1 - val_split) * n_files)):]
    # Datasets, dataloader
    print('Instantiate datasets')
    tc = time.clock()
    tt = time.time()
    train_dataset = dataset(train_list, stack_axis=stax, win_len=win_len, win_hop=win_hop)
    print('train dataset ready')
    val_dataset = dataset(val_list, stack_axis=stax, win_len=win_len, win_hop=win_hop)
    print('Val dataset ready')
    print('CPU time: ' + str(time.clock() - tc) + ' seconds')
    print('User time: ' + str(time.time() - tt) + ' seconds')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    my_model, my_optimizer = load_architecture(archi, n_ch, win_len=win_len)
    # Start afresh if no pretraining, otherwise load pre-trained model and optimizer state dicts
    if state_dicts is None:
        train_losses, val_losses = np.zeros(n_epochs), np.zeros(n_epochs)
        first_epoch, last_epoch = 0, n_epochs
        my_model = my_model.to(device, non_blocking=True)
    else:
        train_losses, val_losses = load_states(my_model, my_optimizer, state_dicts)
        first_epoch, last_epoch = len(train_losses), len(train_losses) + n_epochs
        train_losses = np.concatenate((train_losses, np.zeros(n_epochs)), axis=0)
        val_losses = np.concatenate((val_losses, np.zeros(n_epochs)), axis=0)

    train_callback = SaveAndStop(patience=n_epochs, mode='min')

    # train
    os.makedirs(save_path, exist_ok=True)
    rnd_string = get_model_name(state_dicts)
    print('start training')
    for i_epoch in range(first_epoch, last_epoch):
        train_loss = 0
        val_loss = 0
        for i_batch, (x, y) in enumerate(train_loader):
            train_loss += train_one_batch(my_model,
                                          x.to(device, torch.float32, non_blocking=True),
                                          y.to(device, torch.float32, non_blocking=True),
                                          optimizer=my_optimizer,
                                          output_frames=output_frames)
        with torch.no_grad():   # Gradient will not be computed; Not stored in the graph either.
            for i_batch_val, (x_val, y_val) in enumerate(val_loader):
                val_loss += eval_one_batch(my_model,
                                           x_val.to(device, torch.float32, non_blocking=True),
                                           y_val.to(device, torch.float32, non_blocking=True),
                                           output_frames=output_frames)
        print("epoch {}".format(i_epoch))
        print("\tTrain \t {} \t \t Val \t {}".format(train_loss / len(train_loader), val_loss / len(val_loader)))
        train_losses[i_epoch] = train_loss / len(train_loader)
        val_losses[i_epoch] = val_loss / len(val_loader)

        # Save model and optimizer
        torch.save({'train_loss': train_losses, 'val_loss': val_losses},
                   save_path + '{}_losses.pt'.format(rnd_string))    # Save the losses even if no improvement
        if train_callback.save_model_query(val_losses[i_epoch]):
            print('Saving ' + rnd_string)
            torch.save({
                'model_state_dict': my_model.state_dict(),
                'optimizer_state_dict': my_optimizer.state_dict(),
                'train_loss': train_losses,
                'val_loss': val_losses,
            }, save_path + rnd_string + '_model.pt')
        if train_callback.early_stop_query():
            break
