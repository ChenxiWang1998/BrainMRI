
import torch
import torch.nn as nn

import MetaTransformer
import Module

class MetaBrainInput(nn.Module):
    def __init__(self, name, embedding, dim_z = None):
        super(MetaBrainInput, self).__init__()
        self.name=name
        self.embedding=embedding
        if dim_z:
            self.embedding_vec = nn.parameter.Parameter(torch.zeros(dim_z))
    
    def forward(self, input):
        x = self.embedding(input)
        return x

class MetaBrainOutput(nn.Module):
    def __init__(self, name, predictor):
        super(MetaBrainOutput, self).__init__()
        self.name=name
        self.predictor=predictor
    
    def forward(self, input):
        return self.predictor(input)

class MetaBrainCategoryOutput(MetaBrainOutput):
    def __init__(self, name, predictor):
        super(MetaBrainCategoryOutput, self).__init__(name = name, predictor = predictor)
    
    def forward(self, input):
        pred=super().forward(input)
        if not self.training:
            pred=torch.softmax(pred, dim=1)
        return pred

class ViTConvStack(nn.Module):
    def __init__(self, dim_z = 256, bias = False, norm = "in3d", dropout = 0.2):
        super(ViTConvStack, self).__init__()
        self.dim_z = dim_z
        self.conv = nn.Sequential(
            Module.ConvBlock(1, 8, kernel_size=4, stride=2, padding=1, bias=bias, acfunc=nn.LeakyReLU(0.1)),  #96
            Module.ConvResStack(8, 16, repeats=1, norm=norm, bias=bias, dropout=dropout),  #24
            Module.ConvResStack(16, 32, repeats=2, norm=norm, bias=bias, dropout=dropout),  #24
            Module.ConvResStack(32, 64, repeats=6, norm=norm, bias=bias, dropout=dropout),  #12
            Module.ConvResStack(64, 128, repeats=2, norm=norm, bias=bias, dropout=dropout),  #6
            Module.ConvBlock(128, dim_z, kernel_size= 3, stride = 1, padding = 1, bias = bias, norm = norm, acfunc=None)
        )
    
    def forward(self, inputs):
        x:torch.Tensor = self.conv(inputs)
        x = x.permute(0, 2, 3, 4, 1).view(x.size(0), -1, self.dim_z)
        return x

class MetaBrainViT(nn.Module):
    def __init__(self, dim_z, device, heads, layer_num, dropout):
        super(MetaBrainViT, self).__init__()
        self.dim_z = dim_z
        self.device=device
        self.pe = MetaTransformer.PositionEmbedding(dim_z=dim_z)
        self.input_modules=nn.ModuleDict()
        self.output_modules=nn.ModuleDict()
        
        self.cls = torch.nn.parameter.Parameter(torch.zeros(dim_z, dtype=torch.float32))
        
        self.addInput(MetaBrainInput("mri", ViTConvStack(dim_z = dim_z, dropout = dropout[0]), dim_z=dim_z))
        self.encoder = MetaTransformer.Encoder(dim_z = dim_z, heads = heads, layer_num = layer_num, dropout = dropout[1])
        
        self.to(device)
    
    def resetDevice(self, device):
        if isinstance(device, str):
            device=torch.device(device)
        if isinstance(device, torch.device):
            self.device=device
            self.to(device)
        else:
            print("illegal device:", device)
    
    def addInput(self, input:MetaBrainInput):
        self.input_modules.update({input.name:input.to(self.device)})
    
    def removeInput(self, name):
        del self.input_modules[name]
    
    def addOutput(self, output:MetaBrainOutput):
        self.output_modules.update({output.name:output.to(self.device)})
    
    def removeOutput(self, name):
        del self.output_modules[name]
    
    def clearOutput(self):
        self.output_modules.clear()
    
    def forward(self, inputs:dict, mask:dict):
        batch_size = list(inputs.values())[0].size(0)
        x_list=[self.cls.unsqueeze(0).expand(batch_size, 1, self.dim_z)]
        x_mask_list=[torch.zeros(batch_size, 1, device=self.device).bool()]
        class_list = [torch.zeros(batch_size, 1, self.dim_z, dtype=torch.float32, device = self.device)]
        for n, m in self.input_modules.items():
            embedded_input = m(inputs[n])
            if n != "mri":
                embedded_input = embedded_input.unsqueeze(1) #generate channel dim enssion
            embedded_mask = mask[n].view(-1, 1).expand(batch_size, embedded_input.size(1))#expand mask in channel dimenssion
            x_list.append(embedded_input)
            x_mask_list.append(embedded_mask)
            class_list.append(m.embedding_vec.view(1, 1, -1).expand_as(embedded_input))
        x = torch.concat(x_list, dim=1)
        pe = self.pe(x)
        ce = torch.concat(class_list, dim=1)
        x = nn.functional.instance_norm(x + pe + ce)
        x_mask_list = torch.concat(x_mask_list, dim=1)
        x = self.encoder(x, x_mask_list)
        x = x[:, 0, :]
        x = dict([(name, m(x)) for name, m in self.output_modules.items()]) #calc output
        return x


