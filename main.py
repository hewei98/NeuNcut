import torch
import torch.optim as optim
import numpy as np
import pickle
import argparse
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from models import MLP
from utils import p_normalize, accuracy, same_seeds, full_affinity
from loss import ncut_loss

import os

parser = argparse.ArgumentParser(description='Spectral Classifier')
parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
parser.add_argument('--n_classes', type=int, default=10, help='classes number')
parser.add_argument('--N', type=int, default=20000, help='sample numbers')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--hid_dims', type=list, default=[512, 512], help='hid_dims of NeuNcut')
parser.add_argument('--epo', type=int, default=300, help='training epoch')
parser.add_argument('--bs', type=int, default=1000, help='batch size')
parser.add_argument('--lr', type=float, default=5e-3, help='Adam learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='Adam weight decay')
parser.add_argument('--gamma', type=int, default=80, help='weight of penalty term')
parser.add_argument('--sigma', type=float, default=3.0, help='param of Euclidean distance')
parser.add_argument('--ctn', action='store_true', help='use Continuous penalty function')
parser.add_argument('--step', type=int, default=50, help='update penalty term per step')
parser.add_argument('--p_scale', type=float, default=1.1, help='Penalty term scaler')
parser.add_argument('--g_max', type=int, default=80, help='maximun weight of penalty term')
parser.add_argument('--gpu', type=int, default=0, help='cuda device')
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = '{:d}'.format(args.gpu)
same_seeds(args.seed)
block_size = min(args.N, 10000)

# Load datasets
with open('./datasets/{}/{}_scattering_train_data.pkl'.format(args.dataset,args.dataset), 'rb') as f:
    data_train = pickle.load(f)
with open('./datasets/{}/{}_scattering_test_data.pkl'.format(args.dataset,args.dataset), 'rb') as f:
    data_test = pickle.load(f)
with open('./datasets/{}/{}_scattering_train_label.pkl'.format(args.dataset,args.dataset), 'rb') as f:
    labels_train = pickle.load(f)
with open('./datasets/{}/{}_scattering_test_label.pkl'.format(args.dataset,args.dataset), 'rb') as f:
    labels_test = pickle.load(f)
full_data = np.concatenate([data_train, data_test], axis=0)
full_labels = np.concatenate([labels_train, labels_test], axis=0)
if args.dataset == 'EMNIST':
    print("mean substract applied.")
    full_data = full_data - np.mean(full_data, axis=0, keepdims=True)

full_labels =  full_labels - np.min(full_labels)
sampled_idx = np.random.choice(full_data.shape[0], args.N, replace=False)
data, labels = full_data[sampled_idx], full_labels[sampled_idx]
data = p_normalize(torch.from_numpy(data).float())

# NeuNcut instance
cls_head = MLP(data.shape[1], args.hid_dims, args.n_classes).cuda()

n_iter_per_epoch = args.N // args.bs

optimizer = optim.Adam(cls_head.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epo)

for epoch in range(args.epo):
    randidx = torch.randperm(args.N)
    cls_head.train()
    for i in range(n_iter_per_epoch):
        batch_idx = randidx[i * args.bs : (i + 1) * args.bs]
        batch = data[batch_idx].contiguous().cuda()
        
        # Compute euclidean affinities
        W = full_affinity(X=batch, sigma=args.sigma)

        # Get soft predictions
        P = torch.softmax(cls_head(batch), dim=1)

        # Compute NeuNcut loss
        spectral_loss, orth_reg = ncut_loss(W, P)
        loss = spectral_loss + 0.5 * args.gamma * orth_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if args.ctn:
        if (epoch+1) % args.step==0:
            with torch.no_grad():
                if gamma < args.g_max:
                    gamma = args.p_scale * gamma
    scheduler.step()

    with torch.no_grad():
        cls_head.eval()
        pred = []
        for i in range(args.N // args.bs):
            batch = data[i * args.bs : (i + 1) * args.bs].cuda()
            logits = torch.softmax(cls_head(batch), dim=1)
            batch_pred = torch.argmax(logits, dim=1)
            pred.extend(list(batch_pred.cpu().data.numpy()))
        pred = np.array(pred)
        acc = accuracy(pred, labels)
        print("Epoch{:4d}| ACC:{:.3f}| Trace:{:.3f}| Orth:{:.3f}".format(epoch+1,acc,spectral_loss.item(),orth_reg.item()))


print('evaluating on {}-full...'.format(args.dataset))
full_data = p_normalize(torch.from_numpy(full_data).float()).cuda()
pred = []
for i in range(full_data.shape[0] // 10000):
    batch = full_data[i * 10000 : (i + 1) * 10000].cuda()
    logits = cls_head(batch)
    temp_pred = torch.argmax(logits, dim=1).cpu().data.numpy()
    pred.extend(list(temp_pred))
pred = np.array(pred)
acc = accuracy(pred, full_labels)
nmi = normalized_mutual_info_score(full_labels, pred, average_method='geometric')
ari = adjusted_rand_score(full_labels, pred)
print("ACC:{:3f}| NMI:{:3f}| ARI:{:3f} ".format(acc,nmi,ari))

