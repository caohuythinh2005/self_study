import os
import torch
import numpy as np 

def build_path(path):
    path_levels = path.split('/')
    cur_path = ""
    for path_seg in path_levels:
        if len(cur_path):
            cur_path = cur_path + '/' + path_seg
        else: 
            cur_path = path_seg
        
        if not os.path.exists(cur_path):
            os.mkdir(cur_path)

# [meta | label | feature]

def get_label(data, order, offset, label_dim):
    output = []
    for i in order:
        output.append(data[i][offset:offset+label_dim])

    output = np.array(output, dtype='int')
    return output

def get_feat(data, order, meta_offset, label_dim, feature_dim):
    output = []
    meta_output = []
    offset = meta_offset + label_dim
    for i in order:
        meta_output.append(data[i][:meta_offset])
        output.append(data[i][offset:offset+feature_dim])

    output = np.array(output, dtype='float32')
    meta_output = np.array(meta_output, dtype='float32')
    return np.concatenate([output, meta_output], axis=1)


# dung de tinh log-prob trong truong hop cov la ma tran duong cheo
def log_normal(x, m, v):
    log_prob = (-0.5 * (torch.log(v) + (x-m).pow(2) / v)).sum(-1)
    return log_prob

# 
def log_sum_exp(x, mask):
    x = x.masked_fill(mask == 0, float('-inf'))
    max_x = torch.max(x, 1)[0]
    new_x = x - max_x.unsqueeze(1).expand_as(x)
    return max_x + torch.log(torch.sum(torch.exp(new_x), dim=1) + 1e-10)

def log_mean_exp(x, mask):
    return log_sum_exp(x, mask) - torch.log(mask.sum(1).float() + 1e-10)

def log_normal_mixture(z, m, v, mask=None):
    """
    Tính log-probability của z dưới mô hình Gaussian Mixture.
    
    z: (batch, dim)
    m: (mix, dim)
    v: (mix, dim)
    mask: (batch, mix) hoặc (mix,), optional
    """
    batch, dim = z.shape
    mix = m.shape[0]

    # Expand mean và variance để match batch size
    m = m.unsqueeze(0).expand(batch, -1, -1)  # (batch, mix, dim)
    v = v.unsqueeze(0).expand(batch, -1, -1)
    z = z.unsqueeze(1).expand(batch, mix, dim)

    # Tính log-prob từng component
    indiv_log_prob = log_normal(z, m, v)  # (batch, mix)

    if mask is not None:
        mask = mask.float()
        # Dùng masked_fill để gán -inf cho Gaussian bị mask
        indiv_log_prob = indiv_log_prob.masked_fill(mask == 0, float('-inf'))
    else:
        mask = torch.ones_like(indiv_log_prob)

    # Tính log mean xác suất (với masking)
    log_prob = log_mean_exp(indiv_log_prob, mask)
    return log_prob  # (batch,)




def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True) # [batch_size, 1]
    prods_x = torch.mm(X, X.t()) # [batch_size, batch_size]
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True) # [batch_size, 1]
    prods_y = torch.mm(Y, Y.t()) # [batch_size, batch_size]
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t()) # [batch_size, batch_size]
    dists_c = norms_x + norms_y.t() - 2 * dot_prd # cross distances

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1  = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else: 
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / batch_size
        stats += (res1 - res2)

    return stats
