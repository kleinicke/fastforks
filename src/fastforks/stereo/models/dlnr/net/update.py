import torch
import torch.nn as nn
import torch.nn.functional as F


# fastforks: inference.py runs these modules (the merged LSTM gates, the update order of upstream's
# LSTMMultiUpdateBlock.forward); the unused ConvGRU / SepConvGRU / BasicMultiUpdateBlock are dropped.
class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class LSTM(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(LSTM, self).__init__()
        self.conv_it = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_c_t = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_ft = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_ot = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args

        cor_planes = args.corr_levels * (2 * args.corr_radius + 1)

        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 64, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)


def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


class LSTMMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 128

        self.lstm08 = LSTM(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.lstm16 = LSTM(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.lstm32 = LSTM(hidden_dims[0], hidden_dims[1])
        self.flow_head = FlowHead(hidden_dims[2], hidden_dim=256, output_dim=2)
        factor = 2 ** self.args.n_downsample

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor ** 2) * 9, 1, padding=0))
