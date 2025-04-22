
from math import sqrt

import numpy as np
import torch
import torch.nn as nn

from Module import *

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_z, dim_qk, dim_v, heads, dropout):
        super(MultiHeadSelfAttention, self).__init__()
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.heads = heads
        
        self.scale_factor = sqrt(dim_qk)
        self.query_linear = nn.Linear(dim_z, dim_qk * heads, bias=False)
        self.key_linear = nn.Linear(dim_z, dim_qk * heads, bias=False)
        self.value_linear = nn.Linear(dim_z, dim_v * heads, bias=False)
        
        self.linear = nn.Linear(dim_v * heads, dim_z, bias=False)
        if dropout > 0 and dropout < 1.0:
            self.dropout = nn.Dropout1d(p = dropout)
    
    def forward(self, inputs_query, inputs_key, inputs_value, mask = None):
        batch_size = inputs_query.size(0)
        length_query, length_key, length_value = inputs_query.size(1), inputs_key.size(1), inputs_value.size(1)
        query = self.query_linear(inputs_query).view(batch_size, length_query, self.heads, self.dim_qk).transpose(1, 2)
        key = self.query_linear(inputs_key).view(batch_size, length_key, self.heads, self.dim_qk).transpose(1, 2)
        value = self.query_linear(inputs_value).view(batch_size, length_value, self.heads, self.dim_v).transpose(1, 2)
        alpha = torch.matmul(query, key.transpose(-1, -2)) / self.scale_factor
        if mask is not None:
            alpha.masked_fill_(mask.view(batch_size, 1, 1, length_key), float("-inf"))
        alpha = torch.softmax(alpha, -1)
        x = torch.matmul(alpha, value).transpose(1, 2).contiguous().view((batch_size, length_query, self.heads * self.dim_v))
        x = self.linear(x)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        return x

class PosWiseFFN(nn.Module):
    def __init__(self, dim_z, dropout) -> None:
        super(PosWiseFFN, self).__init__()
        self.linear1 = nn.Linear(dim_z, dim_z)
        self.acfuc = nn.LeakyReLU(0.1)
        self.linear2 = nn.Linear(dim_z, dim_z)
        if dropout > 0 and dropout < 1.0:
            self.dropout = nn.Dropout(p = dropout)
    
    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.acfuc(x)
        x = self.linear2(x)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        return x

class PositionEmbedding(nn.Module):
    def __init__(self, dim_z):
        super(PositionEmbedding, self).__init__()
        self.dim_z = dim_z
    
    def forward(self, inputs):
        pe = torch.from_numpy(PositionEmbedding.position_embedding(length=inputs.size(1), dim_z=self.dim_z)).to(inputs.device)
        return pe
    
    @staticmethod
    def position_embedding(length, dim_z):
        x = 1 / np.power(10000, np.arange(0, dim_z, dtype="f4") / dim_z)
        x = np.stack([x * i for i in range(length)])
        x[:, 0:len(x):2] = np.sin(x[:, 0:len(x):2])
        x[:, 1:len(x):2] = np.cos(x[:, 1:len(x):2])
        return x

class EncoderLayer(nn.Module):
    def __init__(self, dim_z, heads, dropout = 0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadSelfAttention(dim_z=dim_z, dim_qk=dim_z // heads, dim_v = dim_z // heads, heads = heads, dropout=dropout)
        self.ffn = PosWiseFFN(dim_z, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim_z)
        self.norm2 = nn.LayerNorm(dim_z)
    
    def forward(self, encoder_inputs, encoder_mask):
        x = encoder_inputs
        x = self.norm1( x + self.mha(x, x, x, encoder_mask))
        x = self.norm2( x + self.ffn(x))
        return x

class Encoder(nn.Module):
    def __init__(self, dim_z, heads = 8, layer_num = 6, dropout = 0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([ EncoderLayer(dim_z=dim_z, heads=heads, dropout = dropout) for _ in range(layer_num)])
    
    def forward(self, encoder_inputs, encoder_mask):
        x = encoder_inputs
        for layer in self.layers:
            x = layer(x, encoder_mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, dim_z, heads, dropout = 0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadSelfAttention(dim_z=dim_z, dim_qk=dim_z // heads, dim_v = dim_z // heads, heads = heads, dropout = dropout)
        self.cross_attention = MultiHeadSelfAttention(dim_z=dim_z, dim_qk=dim_z // heads, dim_v = dim_z // heads, heads = heads, dropout = dropout)
        self.ffn = PosWiseFFN(dim_z, dropout = dropout)
        self.norm1 = nn.LayerNorm(dim_z)
        self.norm2 = nn.LayerNorm(dim_z)
        self.norm3 = nn.LayerNorm(dim_z)
    
    def forward(self, decoder_inputs, decoder_mask, encoder_outputs, encoder_mask):
        x = decoder_inputs
        x = self.norm1(x + self.self_attention(x, x, x, decoder_mask))
        x = self.norm2(x + self.cross_attention(x, encoder_outputs, encoder_outputs, encoder_mask))
        x = self.norm3(x + self.ffn(x))
        return x

class Decoder(nn.Module):
    def __init__(self, dim_z, heads = 8, layer_num = 6):
        super(Decoder, self).__init__()
        self.layers=nn.ModuleList([DecoderLayer(dim_z, heads = heads) for _ in range(layer_num)])
    
    def forward(self, decoder_inputs, decoder_mask, encoder_outputs, encoder_mask):
        x = decoder_inputs
        for layer in self.layers:
            x = layer(x, decoder_mask, encoder_outputs, encoder_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, dim_z = 256, heads = 8):
        super(Transformer, self).__init__()
        self.dim_z = dim_z
        self.encoder_pe = PositionEmbedding(dim_z=dim_z)
        self.encoder = Encoder(dim_z, heads = heads)
        self.decoder_pe = PositionEmbedding(dim_z=dim_z)
        self.decoder = Decoder(dim_z, heads = heads)
    
    def forward(self, encoder_inputs, decoder_inputs, encoder_mask = None, decoder_mask = None):
        encoder_inputs_pe = encoder_inputs + self.encoder_pe(encoder_inputs) 
        encoder_outputs = self.encoder(encoder_inputs_pe, encoder_mask)
        decoder_inputs_pe = decoder_inputs + self.decoder_pe(decoder_inputs)
        decoder_outputs = self.decoder(decoder_inputs_pe, decoder_mask, encoder_outputs, encoder_mask)
        return decoder_outputs


