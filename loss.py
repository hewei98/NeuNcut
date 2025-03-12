import torch

def ncut_loss(W, P):
    """
    W: N x N affinity graph
    P: N x k predictions
    """
    k = P.shape[1]

    D = torch.diag(torch.sum(W, dim=0))
    L = D - W
    Vol_p = torch.diag(1.0 / torch.sqrt(torch.sum(D.mm(P), dim=0) + 1e-6))
    H = P.mm(Vol_p)
    I_res = H.t().mm(D).mm(H) - torch.eye(k).cuda()

    spectral_loss = torch.trace(H.t().mm(L).mm(H))
    orth_reg = torch.norm(I_res) / k

    return spectral_loss, orth_reg