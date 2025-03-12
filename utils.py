import numpy as np
import torch
from munkres import Munkres
import random


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def regularizer_pnorm(c, p):
    return torch.pow(torch.abs(c), p).sum()


def accuracy(pred, labels):
    err = err_rate(labels, pred)                
    acc = 1 - err
    return acc


def err_rate(gt_s, s):
    """
    Get error rate of the cluster result.
    Fetched from https://github.com/panji1990/Deep-subspace-clustering-networks
    """
    c_x = best_map(gt_s,s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate 


def best_map(L1, L2):
    """ 
    Rearrange the cluster label to minimize the error rate using the Kuhn-Munkres algorithm.
    Fetched from https://github.com/panji1990/Deep-subspace-clustering-networks
    """
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def p_normalize(x, p=2):
    return x / (torch.norm(x, p=p, dim=1, keepdim=True) + 1e-6)


def full_affinity(X, sigma=1.0):
    Y = X
    sum_dims = list(range(2,X.dim()+1))
    X = torch.unsqueeze(X, dim=1)
    distance  = torch.sum(torch.square(X - Y), dim=sum_dims)
    distance = distance / (2 * sigma**2)
    W = torch.exp(-distance)
    return W