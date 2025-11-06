import torch
import torch.nn as nn
from data_utils import min_max_scale, inverse_min_max_scale
import torch.nn.functional as F

Crack_Length = 0
C_value = 1
M_value = 2
porosity = 3



def mse_loss(y_pred, y_true):

    criterion = nn.MSELoss()
    return criterion(y_pred, y_true)



def la_area_loss(y_pred, X_train, clamp_min=0, alpha=0.1):

    dYdX = torch.autograd.grad(
        outputs=y_pred,
        inputs=X_train,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True
    )[0]

    dYdFeature = dYdX[:, porosity]
    positive_grad = torch.clamp(dYdFeature, max=clamp_min)
    la_area = torch.sqrt(torch.sum(positive_grad ** 2))
    return alpha * la_area



def paris_loss(y_pred, X_train,
               m_min, m_max,
               a_min, a_max,
               delta_P=6300,
               B=12,
               W=50,
               eps=1e-6):

    dYdX = torch.autograd.grad(
        outputs=y_pred,
        inputs=X_train,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True
    )[0]
    dNda = dYdX[:, Crack_Length]

    a = X_train[:, Crack_Length].detach()
    C_val = X_train[:, C_value].detach()
    m_val = X_train[:, M_value].detach()

    C_val_orig = 10 ** (C_val.cpu().numpy())
    m_val_orig = inverse_min_max_scale(m_val.cpu().numpy(), m_min, m_max)
    a_val_orig = inverse_min_max_scale(a.cpu().numpy(), a_min, a_max)

    C_val_orig = torch.tensor(C_val_orig, dtype=torch.float32).to(X_train.device)
    m_val_orig = torch.tensor(m_val_orig, dtype=torch.float32).to(X_train.device)
    a_val_orig = torch.tensor(a_val_orig, dtype=torch.float32).to(X_train.device)

    if a_val_orig.shape[0] > 1:
        a_sorted, indices = torch.sort(a_val_orig)
        a_avg = (a_sorted[:-1] + a_sorted[1:]) / 2.0
        alpha_list = (a_avg + 10) / W
        alpha_poly_mean = torch.mean(alpha_list)
    else:
        alpha_poly_mean = (a_val_orig[0] + 10) / W


    poly_coef = (0.886 + 4.64 * alpha_poly_mean -
                 13.32 * alpha_poly_mean ** 2 +
                 14.72 * alpha_poly_mean ** 3 -
                 5.6 * alpha_poly_mean ** 4)


    delta_K = (delta_P * (2 + alpha_poly_mean)) * poly_coef * 0.0316 / (
                B * torch.sqrt(torch.tensor(W, dtype=torch.float32)) * (((1 - alpha_poly_mean) ** (3 / 2)) + eps))




    theory_dNda = 1.0 / (C_val_orig * (delta_K + eps) ** (m_val_orig))

    theory_dNda_np = theory_dNda.cpu().numpy()
    theory_dNda_norm_np, _, _ = min_max_scale(theory_dNda_np)
    theory_dNda_norm = torch.tensor(theory_dNda_norm_np, dtype=torch.float32).to(X_train.device)

    loss = torch.mean((dNda - theory_dNda_norm ) ** 2 )
    return loss



def descending_ranking_loss(y_pred, X, rho_idx, margin=0.0):

    unique_rhos = torch.sort(torch.unique(X[:, rho_idx]))[0]
    means = [ y_pred[X[:, rho_idx] == r].mean() for r in unique_rhos ]
    loss = 0.0
    for i in range(len(means) - 1):

        loss += F.relu(means[i+1] - means[i] + margin)
    return loss


def combined_loss(y_pred, y_true, X_train,
                  m_min, m_max, a_min, a_max,
                  clamp_min=0, alpha=0.1,
                  mse_weight=1.0,
                  paris_weight=0,
                  rank_weight=0.0,
                  margin=0.0,
                  rho_idx=3,
                  ):

    mse_val = mse_loss(y_pred, y_true)
    la_area_val = la_area_loss(y_pred, X_train, clamp_min=clamp_min, alpha=alpha)
    paris_val = paris_loss(
        y_pred,
        X_train,
        m_min, m_max, a_min, a_max
    ) * paris_weight
    rank_loss_val = descending_ranking_loss(y_pred, X_train, rho_idx, margin) * rank_weight

    total_loss = mse_weight *  mse_val + la_area_val+ paris_weight * paris_val+rank_loss_val
    return total_loss, mse_val, la_area_val, paris_val, rank_loss_val
