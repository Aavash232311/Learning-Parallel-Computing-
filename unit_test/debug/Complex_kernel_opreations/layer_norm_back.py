import torch


def layer_backward_analytical(x: torch.Tensor,
                              G: torch.Tensor,
                              gamma: torch.Tensor,
                              mean: torch.Tensor,
                              std: torch.Tensor,
                              B: int,
                              T: int,
                              D: int):
    """
    :param T:
    :param B:
    :param x: Shape(B,T,C)
    :param G: Shape(B,T,C)
    :param gamma: Shape(C,)   i.e. (D,)
    :param mean: Shape(B, T)
    :param std: Shape(B, T)
    :param D: int d_model
    :return: backward tensor shaped (B,T,C)
    """

    epsilon = 1e-8
    mean = mean.reshape(B, T, 1)
    std = std.reshape(B, T, 1)
    gamma = gamma.reshape(1, 1, D)

    D: float = float(D)
    first_comp = 1.0 / (D * std)

    x_hat = (x - mean) / std
    dl_dx_hat = G * gamma

    second_comp = D * dl_dx_hat

    sum_term_1 = dl_dx_hat.sum(dim=-1, keepdim=True)
    sum_term_2 = (dl_dx_hat * x_hat).sum(dim=-1, keepdim=True)

    dx = first_comp * (second_comp - sum_term_1 - x_hat * sum_term_2)
    return dx