import torch
import torch.nn.functional as F
from torch import nn

import pickle

class SamePaddedConv(nn.Module):
    def __init__(self, in_ch, out_ch, padding, kernel_size, **kwargs):
        super().__init__()
        self.pad = padding
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, **kwargs)
    
    def forward(self, x):
        x = F.pad(x, self.pad)
        return self.conv(x)

class AbstractBlock(nn.Module):
    def forward(self, x):
        return self.model(x)

    def _calc_padding(self, resolution, kernel_size, stride):
        if type(resolution) == int:
            width = height = resolution
        elif len(resolution) == 1:
            width = height = resolution[0]
        elif len(resolution) == 2:
            width, height = resolution
        
        o_width, o_height = width // stride, height // stride

        pad_total_width = (o_width - 1) * stride - width + kernel_size
        pad_left = pad_total_width // 2
        pad_right = pad_total_width - pad_left

        pad_total_height = (o_height - 1) * stride - height + kernel_size
        pad_up = pad_total_height // 2
        pad_down = pad_total_height - pad_up

        return (pad_left, pad_right, pad_up, pad_down)

class Depth2Space(AbstractBlock):
    def __init__(self, size):
        super().__init__()
        self.size = size
    
    def forward(self, x):
        b, c, h, w = x.shape
        oh, ow = h * self.size, w * self.size
        oc = c // (self.size * self.size)

        x = torch.reshape(x, (-1, self.size, self.size, oc, h, w))
        x = x.permute(0, 3, 4, 1, 5, 2)
        return torch.reshape(x, (-1, oc, oh, ow))

class Upscale(AbstractBlock):
    def __init__(self, in_ch, out_ch, resolution, scale=2, kernel_size=3, stride=1):
        super().__init__()

        padding = self._calc_padding(resolution, kernel_size, stride)

        self.model = nn.Sequential(
            SamePaddedConv(in_ch, out_ch*scale*scale, padding, kernel_size=kernel_size, stride=1),
            nn.LeakyReLU(0.1),
            Depth2Space(scale)
        )

class Downscale(AbstractBlock):
    def __init__(self, in_ch, out_ch, resolution, kernel_size=3, stride=2):
        super().__init__()

        padding = self._calc_padding(resolution, kernel_size, stride)

        self.model = nn.Sequential(
            SamePaddedConv(in_ch, out_ch, padding, kernel_size=kernel_size, stride=stride),
            nn.LeakyReLU(0.1)
        )

class DownscaleBlock(AbstractBlock):
    def __init__(self, in_ch, ch, resolution, kernel_size=3, stride=2, n_segments=4):
        super().__init__()

        modules = []
        prev_ch = in_ch
        for i in range(n_segments):
            modules.append(Downscale(prev_ch, ch, resolution, kernel_size, stride))
            prev_ch = ch
            ch *= 2
            resolution //= 2
        self.model = nn.Sequential(*modules)

class ResidualBlock(AbstractBlock):
    def __init__(self, in_ch, out_ch, resolution, kernel_size=3):
        super().__init__()
        # No HD version
        
        self.upscale = Upscale(in_ch, out_ch, resolution, kernel_size=kernel_size)
        resolution *= 2

        padding = self._calc_padding(resolution, kernel_size, 1)
        self.model = nn.Sequential(
            SamePaddedConv(out_ch, out_ch, padding, kernel_size=kernel_size),
            nn.LeakyReLU(0.2),
            SamePaddedConv(out_ch, out_ch, padding, kernel_size=kernel_size)
        )
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, inp):
        inp = self.upscale(inp)
        x = self.model(inp)
        return self.activation(x + inp)


class Encoder(AbstractBlock):
    def __init__(self, in_ch, ch, resolution):
        super().__init__()

        self.model = nn.Sequential(
            DownscaleBlock(in_ch, ch, resolution, kernel_size=5),
            nn.Flatten()
        )

    # def load_data(self, saved_model):
    #     downscaleb = self.model[0]
    #     for i in range(4):
    #         downscale = downscaleb.model[i]
    #         downscale.model[0].conv.weight.data = saved_model[f"down1/downs_{i}/conv1/weight:0"]
    #         downscale.model[0].conv.bias.data = saved_model[f"down1/downs_{i}/conv1/bias:0"]

    # def to_pickle(self, path):
    #     dic = {}
    #     downscaleb = self.model[0]
    #     for i in range(4):
    #         downscale = downscaleb.model[i]
    #         dic[f"down1/downs_{i}/conv1/weight:0"] = downscale.model[0].conv.weight.data
    #         dic[f"down1/downs_{i}/conv1/bias:0"] = downscale.model[0].conv.bias.data

    #     with open(path, "wb") as f:
    #         pickle.dump(dic, f)

class Inter(AbstractBlock):
    def __init__(self, in_ch, mid_ch, out_ch, out_res):
        super().__init__()
        self.reshape_dims = (-1, out_ch, out_res, out_res)

        self.linear1 = nn.Linear(in_ch, mid_ch)
        self.linear2 = nn.Linear(mid_ch, out_res * out_res * out_ch)
        self.upscale = Upscale(out_ch, out_ch, out_res)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = torch.reshape(x, self.reshape_dims)
        return self.upscale(x)

    # def load_data(self, saved_model):
    #     self.linear1.weight.data = saved_model["dense1/weight:0"]
    #     self.linear1.bias.data = saved_model["dense1/bias:0"]

    #     self.linear2.weight.data = saved_model["dense2/weight:0"]
    #     self.linear2.bias.data = saved_model["dense2/bias:0"]

    #     self.upscale.model[0].conv.weight.data = saved_model["upscale1/conv1/weight:0"]
    #     self.upscale.model[0].conv.bias.data = saved_model["upscale1/conv1/bias:0"]

    # def to_pickle(self, path):
    #     dic = {}
    #     dic["dense1/weight:0"] = self.linear1.weight.data
    #     dic["dense1/bias:0"] = self.linear1.bias.data

    #     dic["dense2/weight:0"] = self.linear2.weight.data
    #     dic["dense2/bias:0"] = self.linear2.bias.data

    #     dic["upscale1/conv1/weight:0"] = self.upscale.model[0].conv.weight.data
    #     dic["upscale1/conv1/bias:0"] = self.upscale.model[0].conv.bias.data

    #     with open(path, "wb") as f:
    #         pickle.dump(dic, f)

class Decoder(AbstractBlock):
    def __init__(self, in_ch, out_ch, out_mask_ch, resolution):
        super().__init__()

        padding = self._calc_padding(resolution*8, 1, 1)

        self.face_model = nn.Sequential(
            ResidualBlock(in_ch, out_ch*8, resolution),
            ResidualBlock(out_ch*8, out_ch*4, resolution*2),
            ResidualBlock(out_ch*4, out_ch*2, resolution*4),
            SamePaddedConv(out_ch*2, 3, padding, kernel_size=1),
            nn.Sigmoid()
        )

        self.mask_model = nn.Sequential(
            Upscale(in_ch, out_mask_ch*8, resolution),
            Upscale(out_mask_ch*8, out_mask_ch*4, resolution*2),
            Upscale(out_mask_ch*4, out_mask_ch*2, resolution*4),
            SamePaddedConv(out_mask_ch*2, 1, padding, kernel_size=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        f = self.face_model(x)
        m = self.mask_model(x)
        return f, m

    # def load_data(self, saved_model):
    #     for i in range(3):
    #         self.face_model[i].upscale.model[0].conv.weight.data = saved_model[f"upscale{i}/conv1/weight:0"]
    #         self.face_model[i].upscale.model[0].conv.bias.data = saved_model[f"upscale{i}/conv1/bias:0"]

    #         self.face_model[i].model[0].conv.weight.data = saved_model[f"res{i}/conv1/weight:0"]
    #         self.face_model[i].model[0].conv.bias.data = saved_model[f"res{i}/conv1/bias:0"]

    #         self.face_model[i].model[2].conv.weight.data = saved_model[f"res{i}/conv2/weight:0"]
    #         self.face_model[i].model[2].conv.bias.data = saved_model[f"res{i}/conv2/bias:0"]

    #     self.face_model[3].conv.weight.data = saved_model[f"out_conv/weight:0"]
    #     self.face_model[3].conv.bias.data = saved_model[f"out_conv/bias:0"]

    #     for i in range(3):
    #         self.mask_model[i].model[0].conv.weight.data = saved_model[f"upscalem{i}/conv1/weight:0"]
    #         self.mask_model[i].model[0].conv.bias.data = saved_model[f"upscalem{i}/conv1/bias:0"]

    #     self.mask_model[3].conv.weight.data = saved_model[f"out_convm/weight:0"]
    #     self.mask_model[3].conv.bias.data = saved_model[f"out_convm/bias:0"]

    # def to_pickle(self, path):
    #     dic = {}
    #     for i in range(3):
    #         dic[f"upscale{i}/conv1/weight:0"] = self.face_model[i].upscale.model[0].conv.weight.data
    #         dic[f"upscale{i}/conv1/bias:0"] = self.face_model[i].upscale.model[0].conv.bias.data

    #         dic[f"res{i}/conv1/weight:0"] = self.face_model[i].model[0].conv.weight.data
    #         dic[f"res{i}/conv1/bias:0"] = self.face_model[i].model[0].conv.bias.data

    #         dic[f"res{i}/conv2/weight:0"] = self.face_model[i].model[2].conv.weight.data
    #         dic[f"res{i}/conv2/bias:0"] = self.face_model[i].model[2].conv.bias.data

    #     dic[f"out_conv/weight:0"] = self.face_model[3].conv.weight.data
    #     dic[f"out_conv/bias:0"] = self.face_model[3].conv.bias.data

    #     for i in range(3):
    #         dic[f"upscalem{i}/conv1/weight:0"] = self.mask_model[i].model[0].conv.weight.data
    #         dic[f"upscalem{i}/conv1/bias:0"] = self.mask_model[i].model[0].conv.bias.data

    #     dic[f"out_convm/weight:0"] = self.mask_model[3].conv.weight.data
    #     dic[f"out_convm/bias:0"] = self.mask_model[3].conv.bias.data

    #     with open(path, "wb") as f:
    #         pickle.dump(dic, f)

class Model(AbstractBlock):
    def __init__(self, resolution=96, enc_ch=64, int_ch=128, dec_ch=64, dec_mask_ch=64, name=""):
        ## Values taken from Quick96, subject to tuning
        ## Values of SAEHD: resolution=288, enc_ch=92, int_ch=384, dec_ch=72, dec_mask_ch=22
        super().__init__()
        
        self.res = resolution
        self.enc_ch = enc_ch
        self.int_ch = int_ch
        self.dec_ch = dec_ch
        self.dec_mask_ch = dec_mask_ch

        self.enc = Encoder(3, enc_ch, resolution)
        resolution //= 16
        self.inter = Inter(enc_ch * 8 * resolution ** 2, int_ch, int_ch, resolution)
        self.dec_src = Decoder(int_ch, dec_ch, dec_mask_ch, resolution * 2)
        self.dec_dest = Decoder(int_ch, dec_ch, dec_mask_ch, resolution * 2)

    def forward(self, x, kind):
        if kind == "DEST":
            dec = self.dec_dest
        elif kind == "SRC":
            dec = self.dec_src
        
        x = self.enc(x)
        x = self.inter(x)
        return dec(x)

    def get_parameters(self):
        return { "resolution": self.res, 
                 "enc_ch": self.enc_ch, 
                 "int_ch": self.int_ch, 
                 "dec_ch": self.dec_ch, 
                 "dec_mask_ch": self.dec_mask_ch }