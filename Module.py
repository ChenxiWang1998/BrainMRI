import torch
import torch.nn as nn

import MetaBrain

class PixelNorm(nn.Module):
    def __init__(self, sigma=1e-8):
        super(PixelNorm, self).__init__()
        self.sigma=sigma
    
    def forward(self, inputs:torch.Tensor):
        x=inputs
        mean=x.pow(2).mean(dim=1, keepdim=True).sqrt()
        print(mean)
        return x / mean

class VarianceInstanceNorm(nn.Module):
    def __init__(self, dim):
        super(VarianceInstanceNorm, self).__init__()
        self.dim=dim
    
    def forward(self, input:torch.Tensor):
        dim = list(range(len(input.size()) - 1, len(input.size()) - 1 - self.dim, -1))
        var=input.pow(2).mean(dim = dim, keepdim=True).sqrt()
        x=input/var
        return x

class VarianceInstanceNorm1d(VarianceInstanceNorm):
    def __init__(self):
        super(VarianceInstanceNorm1d, self).__init__(1)

class VarianceInstanceNorm2d(VarianceInstanceNorm):
    def __init__(self):
        super(VarianceInstanceNorm2d, self).__init__(2)

class VarianceInstanceNorm3d(VarianceInstanceNorm):
    def __init__(self):
        super(VarianceInstanceNorm3d, self).__init__(3)

class VarianceNorm(nn.Module):
    def __init__(self, feature_num):
        super(VarianceNorm, self).__init__()
        self.feature_num = feature_num
        self.weight = nn.parameter.Parameter(torch.ones(1, feature_num, 1, 1, 1))
    
    def forward(self, input):
        var=input.pow(2).mean(dim = (-1, -2, -3, -4), keepdim=True).sqrt()
        x=input/var
        x = x * self.weight
        return x

class VarianceNorm2(nn.Module):
    def __init__(self, feature_num):
        super(VarianceNorm2, self).__init__()
        self.feature_num = feature_num
        # self.weight = nn.parameter.Parameter(torch.ones(1, feature_num, 1, 1, 1))
    
    def forward(self, input):
        var=input.pow(2).mean(dim = (-1, -2, -3), keepdim=True).sqrt()
        x=input/var
        # x = x * self.weight
        return x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=0.00001):
        super(LayerNorm, self).__init__()
        self.dim = dim
        self.eps=eps
    
    def forward(self, input):
        mean = torch.mean(input, dim=self.dim, keepdim=True)
        std = torch.std(input, dim=self.dim, keepdim=True) + self.eps
        # print(mean.size(), std.size())
        x = (input - mean) / std
        return x

def normalLayer(norm:str, num_features:int=None):
    assert norm in ["in3d", "in2d", "in1d", "pn", "vin3d", "vin2d", "vin1d", "bn3d", "bn2d", "bn1d", "vn", "vn2", "ln1d", "ln2d", "ln3d"], "illigal symble: %s" % norm
    if (norm == "in3d"):
        return nn.InstanceNorm3d(num_features, affine=True)
    if (norm == "in2d"):
        return nn.InstanceNorm2d(num_features)
    if (norm == "in1d"):
        return nn.InstanceNorm1d(num_features)
    if (norm == "pn"):
        return PixelNorm()
    if (norm == "vin3d"):
        return VarianceInstanceNorm3d()
    if (norm == "vin2d"):
        return VarianceInstanceNorm2d()
    if (norm == "vin1d"):
        return VarianceInstanceNorm1d()
    if (norm == "bn3d"):
        return nn.BatchNorm3d(num_features)
    if (norm == "bn2d"):
        return nn.BatchNorm2d(num_features)
    if (norm == "bn1d"):
        return nn.BatchNorm1d(num_features)
    if (norm == "vn"):
        return VarianceNorm(num_features)
    if (norm == "vn2"):
        return VarianceNorm2(num_features)
    if (norm == "ln1d"):
        return LayerNorm((-1, -2))
    if (norm == "ln2d"):
        return LayerNorm((-1, -2, -3))
    if (norm == "ln3d"):
        return LayerNorm((-1, -2, -3, -4))
    return None

class LinearLayer(nn.Sequential):
    def __init__(self, in_features, out_features, dropout=None, acfunc=nn.LeakyReLU(0.1), norm=None):
        super(LinearLayer, self).__init__()
        if(dropout):
            self.add_module("dropout", nn.Dropout(p=dropout))
        self.add_module("linear", nn.Linear(in_features, out_features))
        if(norm is not None):
            self.add_module(norm, normalLayer(norm, out_features))
        if(acfunc):
            self.add_module("acfunc", acfunc)

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, norm=None, acfunc=None):
        super().__init__()
        self.add_module("conv", nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        if(norm): #是否添加标准化层
            name="norm"
            if(isinstance(norm, str)):
                name=norm
                norm=normalLayer(norm, out_channels)
            assert isinstance(norm, nn.Module), "norm can only be nn.Module or string!"
            self.add_module(name, norm)
        if(acfunc is not None): #是否添加激活函数
            self.add_module("activation", acfunc)

class ConvTransBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, norm=None, acfunc=None):
        super().__init__()
        self.add_module("conv_transpose", nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        if(norm): #是否添加标准化层
            name="norm"
            if(isinstance(norm, str)):
                name=norm
                norm=normalLayer(norm, out_channels)
            assert isinstance(norm, nn.Module), "norm can only be nn.Module or string!"
            self.add_module(name, norm)
        if(acfunc is not None): #是否添加激活函数
            self.add_module("activation", acfunc)

class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dropout=None, norm=None):
        super(ConvResBlock, self).__init__()
        self.convblock1=ConvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, norm=norm, acfunc=nn.LeakyReLU(0.1))
        self.convblock2=ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout=nn.Dropout3d(dropout) if(dropout) else None
        self.acfunction=nn.LeakyReLU(0.1)
        if(isinstance(norm, str)):
            self.norm=normalLayer(norm, out_channels)
        self.downsample=None
        if(stride!=1 or in_channels != out_channels):
            self.downsample=ConvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, acfunc=None)
    
    def forward(self, inputs):
        x=inputs
        x=self.convblock1(x)
        if(self.dropout):
            x=self.dropout(x)
        x=self.convblock2(x)
        if(self.dropout):
            x=self.dropout(x)
        identity=self.downsample(inputs) if(self.downsample) else inputs
        x=x+identity
        if(hasattr(self, "norm")):
            x=self.norm(x)
        x=self.acfunction(x)
        return x

class ConvResStack(nn.Sequential):
    def __init__(self, in_channels, out_channels, repeats, norm, bias=True, down_sample_first=True, dropout = None):
        super(ConvResStack, self).__init__()
        if(down_sample_first):
            self.add_module("down_sample", ConvResBlock(in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm=norm, bias=bias, dropout=dropout))
            for idx in range(1, repeats+1):
                self.add_module("conv%d"%idx, ConvResBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, norm=norm, bias=bias, dropout=dropout))
        else:
            for idx in range(1, repeats+1):
                self.add_module("conv%d"%idx, ConvResBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, norm=norm, bias=bias, dropout=dropout))
            self.add_module("down_sample"%repeats, ConvResBlock(in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm=norm, bias=bias, dropout=dropout))

class ConvModule(nn.Sequential):
    def __init__(self, z_dim = 256, norm="in3d", bias=True, dropout = None):
        super(ConvModule, self).__init__()
        self.add_module("stack1", ConvBlock(1, 8, kernel_size=4, stride=2, padding=1, bias=bias, acfunc=nn.LeakyReLU(0.1))) #96
        self.add_module("stack2", ConvResStack(8, 16, repeats=1, norm=norm, bias=bias, dropout=dropout)) #48
        self.add_module("stack3", ConvResStack(16, 32, repeats=2, norm=norm, bias=bias, dropout=dropout)) #24
        self.add_module("stack4", ConvResStack(32, 64, repeats=3, norm=norm, bias=bias, dropout=dropout)) #12
        self.add_module("stack5", ConvResStack(64, 128, repeats=3, norm=norm, bias=bias, dropout=dropout)) #6
        self.add_module("stack6", ConvResStack(128, z_dim, repeats=3, norm=norm, bias=bias, dropout=dropout)) #3
        self.add_module("avg", nn.AvgPool3d(kernel_size=3))
        self.add_module("flatten", nn.Flatten())

class ConvModuleVGG(nn.Sequential):
    def __init__(self, z_dim = 256, norm="in3d", bias=True):
        super(ConvModuleVGG, self).__init__()
        self.add_module("stack1", ConvBlock(1, 8, kernel_size=4, norm = norm, bias=bias, stride=2, padding=1, acfunc=nn.LeakyReLU(0.1))) #96
        self.add_module("stack2", ConvResBlock(8, 16, kernel_size=4, stride=2, padding=1, norm=norm, bias=bias)) #48
        self.add_module("stack3", ConvResBlock(16, 32, kernel_size=4, stride=2, padding=1, norm=norm, bias=bias)) #24
        self.add_module("stack4", ConvResBlock(32, 64, kernel_size=4, stride=2, padding=1, norm=norm, bias=bias)) #12
        self.add_module("stack5", ConvResBlock(64, 128, kernel_size=4, stride=2, padding=1, norm=norm, bias=bias)) #6
        self.add_module("stack6", ConvResBlock(128, z_dim, kernel_size=4, stride=2, padding=1, norm=norm, bias=bias)) #3
        self.add_module("avg", nn.AvgPool3d(kernel_size=3))
        self.add_module("flatten", nn.Flatten())

class ConvModule_Mini(nn.Sequential):
    def __init__(self, z_dim=256, norm="in3d", bias=True):
        super(ConvModule_Mini, self).__init__()
        self.add_module("stack1", ConvBlock(1, 8, kernel_size=4, stride=2, padding=1, bias=bias, acfunc=nn.LeakyReLU(0.1))) #96
        self.add_module("stack2", ConvResStack(8, 16, repeats=1, norm = norm, bias = bias)) #48
        self.add_module("stack3", ConvResStack(16, 32, repeats=2, norm = norm, bias = bias)) #24
        self.add_module("stack4", ConvResStack(32, 64, repeats=3, norm = norm, bias = bias)) #12
        self.add_module("stack5", ConvResStack(64, z_dim, repeats=3, norm = norm, bias = bias)) #6
        self.add_module("avg", nn.AvgPool3d(kernel_size=3))
        self.add_module("flatten", nn.Flatten())

def generateAgeInput(name="age", dim_z = 256):
    return MetaBrain.MetaBrainInput(name, embedding=nn.Sequential(
        LinearLayer(1, 256,), 
        nn.Linear(256, dim_z)
    ), dim_z=dim_z)

def generateSexInput(name="sex", dim_z = 256):
    return MetaBrain.MetaBrainInput(name, embedding=nn.Sequential(
        nn.Linear(2, dim_z)
    ), dim_z=dim_z)

def generateAPOEInput(name="APOE", dim_z = 256):
    return MetaBrain.MetaBrainInput(name, embedding=nn.Sequential(
        nn.Linear(6, dim_z)
    ), dim_z=dim_z)

def generatePredictor(num, dim_z):
    return nn.Sequential(
        LinearLayer(dim_z, dim_z, dropout=0.2, norm="in1d"), 
        LinearLayer(dim_z, dim_z, dropout=0.2), 
        nn.Linear(dim_z, num)
    )

class IndexSelector(nn.Module):
    def __init__(self, *index):
        super(IndexSelector, self).__init__()
        self.register_buffer("index", torch.tensor(index, dtype=torch.int32))
    
    def forward(self, input):
        return torch.index_select(input, dim=1, index=self.index)


