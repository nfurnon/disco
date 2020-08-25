import torch


def nanmean(v, *args, inplace=False, **kwargs):
    """Compute the mean ignoring NaN values.
    Credits: https://github.com/yulkang @ https://github.com/pytorch/pytorch/issues/21987#issuecomment-539402619
    """
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def reconstruction_loss(y_true, y_pred, y_in):
    """ MSE on mask applied on input STFT.
    Args:
        y_true (tensor): ground truth
        y_pred (tensor): predicted tensor
        y_in (tensor): tensor of same shape as y_true and y_pred, frame(s) of
                       the input spectrograms to apply the mask on.
    Returns:
        (float): A loss function value, the MSE weighted by the input STFT.
    """
    return nanmean(((y_pred - y_true) * y_in) ** 2)
