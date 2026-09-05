import torch


def layer_backward_analytical(x:torch.Tensor,
                              G:torch.Tensor,
                              gamma:torch.Tensor,
                              mean:torch.Tensor,
                              std:torch.Tensor,
                              B:int,
                              T:int,
                              D: int):
    """
    :param x: Shape(B,T,C)
    :param G: Shape(B,T,C)
    :param gamma: Shape(B, 1, 1,)
    :param mean: Shape(B, T)
    :param std: Shape(B, T)
    :param D: int d_model
    :return: backward tensor shaped (B,T, C)
    """
    mean = mean.reshape(B, T, 1)
    std = std.reshape(B, T, 1)

    print(f"mean shape: {mean.shape} \n"
          f"G shape: {G.shape} \n"
          f"std shape: {std.shape} \n"
          f"gamma shape: {gamma.shape} \n"
          f"x shape: {x.shape} \n")

    D: float = float(D)
    first_comp = 1.0 / (D * std)

    x_hat = (x - mean) / std  # (B, T, C)
    dl_dx_hat = G * gamma # (B, T, C)

    second_comp = D * x_hat

    # again row wise sum there for both sum component
    sum_term_1 = dl_dx_hat.sum(dim=-1, keepdim=True)  # (B, T, 1)
    sum_term_2 =  (dl_dx_hat * x_hat).sum(dim=-1, keepdim=True) # (B, T, 1)

    dx = first_comp * (second_comp - sum_term_1 - x_hat * sum_term_2)
    return dx

