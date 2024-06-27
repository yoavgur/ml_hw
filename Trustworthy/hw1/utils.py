import gzip
import struct
from os import path
import numpy as np
import models
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def load_pretrained_cnn(cnn_id, n_classes=4, models_dir='trained-models/'):
    """
    Loads one of the pre-trained CNNs that will be used throughout the HW
    """
    if not isinstance(cnn_id, int) or cnn_id < 0 or cnn_id > 2:
        raise ValueError(f'Unknown cnn_id {id}')
    model = eval(f'models.SimpleCNN{cnn_id}(n_classes=n_classes)')
    fpath = path.join(models_dir, f'simple-cnn-{cnn_id}')
    model.load_state_dict(torch.load(fpath))
    return model


class TMLDataset(Dataset):
    """
    Used to load the dataset used throughout the HW
    """

    def __init__(self, fpath='dataset.npz', transform=None):
        with gzip.open(fpath, 'rb') as fin:
            self.data = np.load(fin, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def compute_accuracy(model, data_loader, device):
    """
    Evaluates and returns the (benign) accuracy of the model 
    (a number in [0, 1]) on the labeled data returned by 
    data_loader.
    """
    
    total = 0
    correct = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total

def run_whitebox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the white-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=c_x+randint(1, n_classes)%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    """
    x_advs, ys = [], []

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        if targeted:
            y = ((y + torch.randint(1, n_classes, (y.size(0),)).to(device)) % n_classes).to(device)

        x_adv, y = attack.execute(x, y, targeted)
        x_advs.append(x_adv)
        ys.append(y)

    return torch.cat(x_advs), torch.cat(ys)


def run_blackbox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the black-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=(c_x+randint(1, n_classes))%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    3- The number of queries made to create each adversarial example.
    """
    x_advs, ys, n_queriess = [], [], []

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        if targeted:
            y = ((y + torch.randint(1, n_classes, (y.size(0),)).to(device)) % n_classes).to(device)

        x_adv, n_queries = attack.execute(x, y, targeted)

        x_advs.append(x_adv)
        ys.append(y)
        n_queriess.append(n_queries)

    return torch.cat(x_advs), torch.cat(ys), torch.cat(n_queriess)


def compute_attack_success(model, x_adv, y, batch_size, targeted, device):
    """
    Returns the success rate (a float in [0, 1]) of targeted/untargeted
    attacks. y contains the true labels in case of untargeted attacks,
    and the target labels in case of targeted attacks.
    """
    with torch.no_grad():
        x_adv = x_adv.to(device)
        preds = model(x_adv).argmax(dim=1)

        if targeted:
            return (preds == y).float().mean().item()
        else:
            return (preds != y).float().mean().item()




def binary(num):
    """
    Given a float32, this function returns a string containing its
    binary representation (in big-endian, where the string only
    contains '0' and '1' characters).
    """
    return ''.join(format(struct.unpack('!I', struct.pack('!f', num))[0], '032b'))


def float32(binary):
    """
    This function inverts the "binary" function above. I.e., it converts 
    binary representations of float32 numbers into float32 and returns the
    result.
    """
    return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]


def random_bit_flip(w):
    """
    This functoin receives a weight in float32 format, picks a
    random bit to flip in it, flips the bit, and returns:
    1- The weight with the bit flipped
    2- The index of the flipped bit in {0, 1, ..., 31}
    """
    binary_rep = binary(w)
    idx = np.random.choice(32)
    flipped = '1' if binary_rep[idx] == '0' else '0'
    binary_rep = binary_rep[:idx] + flipped + binary_rep[idx+1:]
    return float32(binary_rep), idx
