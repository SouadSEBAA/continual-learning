import copy
import random
import torch

from torch.utils.data import Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.targets = [self.dataset.targets[idx] for idx in self.idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]

    def get_unique_targets(self):
        return list(set(self.targets))


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def fl_exp_details(
    iid: bool,
    num_clients: int,
    frac: int,
    local_batch_size: int,
    local_iters: int,
    global_iters: int,
):
    print("    Federated parameters:")
    if iid:
        print("    IID")
    else:
        print("    Non-IID")
    print(f"    Number of clients  : {num_clients}")
    print(f"    Fraction of clients  : {frac}")
    print(f"    Local Batch size   : {local_batch_size}")
    print(f"    Local Epochs       : {local_iters}\n")
    print(f"    Global Epochs       : {global_iters}\n")
    return


def distribution(minval, maxval, numclients, roundnum=2):
    l = [round(random.uniform(minval, maxval), roundnum) for _ in range(numclients)]
    return [round(x / sum(l), roundnum) for x in l]
