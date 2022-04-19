from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class BinaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k, t, sparse_ratio_):
        ctx.save_for_backward(input, k, t)
        out = input.new(input.size())
        out[input >= 0] = 1
        out[input < 0] = 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t * 2), 2)) * grad_output
        return grad_input, None, None, None


class PUERT(torch.nn.Module):
    def __init__(self, LayerNo, rb_num, desired_sparsity, sparse_ratio_=0, flag_1D=False):
        super(PUERT, self).__init__()
        self.desired_sparsity = desired_sparsity
        self.sparse_ratio_ = sparse_ratio_
        self.flag_1D = flag_1D

        self.MyBinarize = BinaryQuantize.apply
        self.k = torch.tensor([10]).float().to(device)
        self.t = torch.tensor([0.1]).float().to(device)
        self.mask_shape = (256, 256) if not flag_1D else (256)
        self.pmask_slope = 5

        self.Phi = nn.Parameter(self.initialize_p())

        onelayer = []
        self.LayerNo = LayerNo
        basicblock = ISTA_2RB_BasicBlock
        for i in range(LayerNo):
            onelayer.append(basicblock(rb_num=rb_num))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, gt):
        maskp0 = torch.sigmoid(self.pmask_slope * self.Phi)

        maskpbar = torch.mean(maskp0)
        r = self.desired_sparsity / maskpbar
        beta = (1 - self.desired_sparsity) / (1 - maskpbar)
        le = torch.le(r, 1).float()
        maskp = le * maskp0 * r + (1 - le) * (1 - (1 - maskp0) * beta)

        u = torch.from_numpy(np.random.uniform(low=0.0, high=1.0, size=maskp0.size())).type(dtype)

        mask_matrix = self.MyBinarize(maskp - u, self.k, self.t, self.sparse_ratio_)

        if self.flag_1D:
            mask_matrix_1D = mask_matrix.unsqueeze(-1)
            mask_matrix = torch.cat([mask_matrix_1D] * 256, dim=1)

        mask = mask_matrix.unsqueeze(0).unsqueeze(-1)
        xu_real = zero_filled(gt, mask)

        x = xu_real

        for i in range(self.LayerNo):
            x = self.fcs[i](x, xu_real, mask)

        x_final = x

        return [x_final, mask_matrix, maskp]

    def initialize_p(self, eps=0.01):
        x = torch.from_numpy(np.random.uniform(low=eps, high=1-eps, size=self.mask_shape)).type(dtype)
        return - torch.log(1. / x - 1.) / self.pmask_slope

class ISTA_2RB_BasicBlock(torch.nn.Module):
    def __init__(self, rb_num):
        super(ISTA_2RB_BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        kernel_size = 3
        bias = True
        n_feat = 32

        self.conv_D = nn.Conv2d(1, n_feat, kernel_size, padding=(kernel_size//2), bias=bias)

        modules_body = [Residual_Block(n_feat, n_feat, 3, bias=True, res_scale=1) for _ in range(rb_num)]

        self.body = nn.Sequential(*modules_body)

        self.conv_G = nn.Conv2d(n_feat, 1, kernel_size, padding=(kernel_size//2), bias=bias)

    def forward(self, x, PhiTb, mask):
        x = x - self.lambda_step * zero_filled(x, mask)
        x = x + self.lambda_step * PhiTb
        x_input = x

        x_D = self.conv_D(x_input)

        x_backward = self.body(x_D)

        x_G = self.conv_G(x_backward)

        x_pred = x_input + x_G

        return x_pred

class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, res_scale=1):

        super(Residual_Block, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.act1 = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        res = x
        x = res * self.res_scale + input
        return x
