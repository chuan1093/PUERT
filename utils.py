import torch.nn as nn
import torch
import numpy as np
import math

def zero_filled(x, mask, mod=False, norm=False):
    '''
    :param x: tensor of size (bs, C, H, W), C=1 for real image, C=2 for complex image
    :param mask: 1,H,W,1
    '''
    x_dim_0 = x.shape[0]
    x_dim_1 = x.shape[1]
    x_dim_2 = x.shape[2]
    x_dim_3 = x.shape[3]
    x = x.view(-1, x_dim_2, x_dim_3, 1)

    x_real = x
    x_imag = torch.zeros_like(x_real)
    x_complex = torch.cat([x_real, x_imag], 3)

    x_kspace = torch.fft(x_complex, 2, normalized=norm)
    y_kspace = x_kspace * mask
    xu = torch.ifft(y_kspace, 2, normalized=norm)

    if not mod:
        xu_ret = xu[:, :, :, 0:1]
    else:
        xu_ret = torch.sqrt(xu[..., 0:1] ** 2 + xu[..., 1:2] ** 2)

    xu_ret = xu_ret.view(x_dim_0, x_dim_1, x_dim_2, x_dim_3)

    return xu_ret


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def get_parameter_number(net):
    return sum(p.numel() for p in net.parameters())


def show_model_paras(model, flag=True):
    if flag:
        num_count = 0
        for para in model.parameters():
            num_count += 1
            print('Layer %d' % num_count)
            print(para.size())
    print('Total parameters: %d' % (get_parameter_number(model)))


def get_optimizer(optimizer_type, param, lr, weight_decay):
    if optimizer_type == 'RMSprop':
        optimizer = torch.optim.RMSprop(param, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(param, lr=lr, weight_decay=weight_decay)
    else:
        assert False, "Wrong optimizer type"
    return optimizer
