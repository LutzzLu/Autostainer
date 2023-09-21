import torch
import torch.nn.functional as F

def tp_tn_fp_fn(pred, true):
    pred = pred > 0.5
    true = true > 0.5
    tp = (pred & true).sum(dim=0).float()
    tn = (~pred & ~true).sum(dim=0).float()
    fp = (pred & ~true).sum(dim=0).float()
    fn = (~pred & true).sum(dim=0).float()
    return tp, tn, fp, fn

def balanced_binary(input, target):
    pos = target == 1
    neg = target == 0
    pos_weight = neg.sum() / pos.sum()
    return F.binary_cross_entropy_with_logits(input, target, pos_weight=pos_weight.unsqueeze(0))

def ZINB_loss(
    x,
    log_mean,
    disp,
    pi,
    # scale_factor=1.0,
    ridge_lambda=0.0
):
    eps = 1e-10

    mean_exp = torch.clamp(torch.exp(log_mean), min=1e-5, max=1e6)

    t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
    t2 = (disp+x) * torch.log(1.0 + (mean_exp/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean_exp+eps)))
    nb_final = t1 + t2

    nb_case = nb_final - torch.log(1.0-pi+eps)
    zero_nb = torch.pow(disp/(disp+mean_exp+eps), disp)
    zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
    result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

    if ridge_lambda > 0:
        ridge = ridge_lambda*torch.square(pi)
        result += ridge
    result = torch.mean(result)
    return result