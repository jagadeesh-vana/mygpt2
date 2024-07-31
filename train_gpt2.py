# Import necessary libraries
import math
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
        super().__init__()
        # Ensure the number of embedding dimensions (n_embd) is divisible by the number of heads (n_head),
        # which is a requirement for multi-head attention.
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # A linear layer that projects the input into query, key, and value vectors. 
        # The output dimension is 3 * config.n_embd because we need separate vectors for queries, keys, and values.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        #A linear layer that projects the concatenated output of the attention heads back to the 
        # original embedding size.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.MYGPT2_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        # A lower triangular matrix used as a causal mask to prevent attending to future tokens.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        #self.register_buffer is a method provided by PyTorch's nn.Module class, which is used to register
        # a tensor as a buffer in the module. Buffers are similar to parameters, but they are
        # non-trainable and persistent.
        # self.register_buffer(name, tensor) -> name (str): The name of the buffer, tensor (Tensor): The tensor to register as a buffer.

    def forward(self, x):   
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x) #  Projects input x into a combined query, key, and value vector of shape (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embd, dim=2) # Splits the combined vector into three parts: query (q), key (k), and value (v), each of shape (B, T, C)
        # split() method splits a tensor into multiple sub-tensors along a specified dimension.
        
        # Reshape for multi head attention
        # Reshapes q, k, and v to separate the heads and then transposes to bring the head dimension forward, 
        # resulting in shape (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        
        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # Applies the causal mask to prevent attending to future tokens.
        #att = F.softmax(att, dim=-1) # Applies the softmax function to obtain the attention weights.
        # Computes the weighted sum of values based on the attention weights, resulting in shape (B, nh, T, hs).
        #y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Flash Attention is 7.6x times faster because of kernel infusion of the above 4 steps.
        y = F.scaled_dot_product_attention(q, k, v, is_causal = True)
        
        #Transposes and reshapes the output to combine the heads back into the original embedding size.
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        # self.c_proj(y) applies the final linear layer to project the concatenated output back to the embedding size.
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.MYGPT2_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight   
        
        # init params
        self.apply(self.__init_weights)
        
    def __init_weights(self, module):        
        
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'MYGPT2_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss  = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # @classmethod is a decorator used to define a method that is bound to the class and not the 
    # instance of the class. This means that the method can be called on the class itself, 
    # rather than on instances of the class.
    # Bound to the Class: A class method receives the class as its first argument, conventionally 
    # named cls, which stands for class. This is similar to instance methods, which receive the 
    # instance as their first argument, conventionally named self
    # Class methods can access and modify class variables that are shared among all instances of the class.
    # Class methods do not have access to instance variables or methods directly unless the instance is passed to them.
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# -----------------------------------------------------------------------------
#model = GPT.from_pretrained('gpt2')
#print("didn't crash yay!")

#-----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 
import tiktoken

class DataLoaderLite:
    
    def __init__(self, B, T):
        self.B = B
        self.T = T
    
        # At init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")
        
        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # If loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y 

#-----------------------------------------------------------------------------
# attempt to autodetect the device

import time

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

train_loader = DataLoaderLite(B=4, T=1024)

torch.set_float32_matmul_precision('high') # high means float32.
# A100 - the Ampeare series GPU will run 8x faster converting the calculations to TF32 inside it's hardware.

# get logits
model = GPT(GPTConfig(vocab_size = 50304))  # randomly initialized model
# making vocab size a nice number that is divisible by 2, 4, 8, 16, 32 and so on.
model.to(device)
model = torch.compile(model) # Speedup mainly comes from reducing python overheads (no interpreter) and GPU read/writes

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(10):
    t0 = time.time()
    # The time.time() function in Python returns the current time in seconds since the Epoch
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type = device, dtype=torch.bfloat16): # converting float32 to bfloats for faster calculations
        logits, loss = model(x, y)
    # import code; code.interact(local=locals())
    # The code.interact function from Python's code module provides an interactive console that you can use to 
    #inspect and manipulate the current local variables and environment. This can be particularly useful for debugging or exploring code.
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize() # CPU will wait for the GPU to finish running, then we can take time.
    t1 = time.time()
    dt = (t1 - t0) * 1000 # time difference in milli seconds
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {i}, loss : {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec}")

import sys; sys.exit(0)

# Generate from the model
num_return_sequences = 5
max_length = 30

#model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())  # randomly initialized model
model.eval()
model.to(device)

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I am a language model")
tokens = torch.tensor(tokens, dtype = torch.long) # (8, 1)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5,8)
x = tokens.to(device)

# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)