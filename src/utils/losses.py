import torch
from torch import nn
from torch.autograd import grad


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0][:, -3:]
    return points_grad


def total_variation_loss(tensor):
    bs, c, h, w, d = tensor.size()
    tv_h = torch.pow(tensor[:, :, 1:, :, :] - tensor[:, :, :-1, :, :], 2).sum()
    tv_w = torch.pow(tensor[:, :, :, 1:, :] - tensor[:, :, :, :-1, :], 2).sum()
    tv_d = torch.pow(tensor[:, :, :, :, 1:] - tensor[:, :, :, :, :-1], 2).sum()
    return (tv_h + tv_w + tv_d) / (bs * c * h * w * d)


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, pred, gt):
        smooth = 1e-6
        intersection = pred * gt
        union = pred + gt - intersection
        loss = (intersection / (union + smooth)).mean((1, 2, 3))
        loss = 1 - loss
        return loss
