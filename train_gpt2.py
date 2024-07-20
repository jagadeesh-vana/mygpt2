# Import necessary libraries
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
#-----------------------------

# Let's load gpt2 124M model into our class here. 
# GPT2 is slightly modified from the original Transformer. In particular, we
# did not have encoder and cross attention blocks and is a decoder only architecture.
# In the decoder block, layer normalization is moved to the input of each sub-block
# and an additional layer normalization was added after the final self attention block.

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        pass
    

class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
        

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) # Layernorm before input to attention
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)  # Layernorm before input to FF
        self.MLP  = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.MLP(self.ln_2(x))
        return x

@dataclass
class GPTConfiguration:
    block_size: int = 256
    vocab_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    
class GPTSmall(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # input embd
            wpe  = nn.Embedding(config.block_size, config.n_embd), # pos embd
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # attention and ff
            ln_f = nn.layerNorm(config.n_embd), # additional layer norm
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # last linear layer
        
        