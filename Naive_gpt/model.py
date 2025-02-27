# Model/model.py
import torch
import torch.nn as nn
from .config import *
import inspect


class CausalSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()
        assert N_EMBD % N_HEAD == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD)
        # output projection
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = N_HEAD
        self.n_embd = N_EMBD

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        y = nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True)  # flash attention
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)
        return y
    
# class Head(nn.Module):  #this is sebastian's causal attention
#     """ one head of self-attention """

#     def __init__(self, head_size):
#         super().__init__()
#         self.key = nn.Linear(N_EMBD, head_size, bias=False)
#         self.query = nn.Linear(N_EMBD, head_size, bias=False)
#         self.value = nn.Linear(N_EMBD, head_size, bias=False)
#         self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
#         self.dropout = nn.Dropout(DROPOUT)

#     def forward(self, x):
#         B, T, C = x.shape
#         k = self.key(x)   # (B, T, head_size)
#         q = self.query(x) # (B, T, head_size)
#         wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
#         wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
#         wei = nn.functional.softmax(wei, dim=-1)
#         wei = self.dropout(wei)
#         v = self.value(x)
#         out = wei @ v
#         return out

# class MultiHeadAttention(nn.Module):
#     """ multiple heads of self-attention in parallel """

#     def __init__(self, num_heads, head_size):
#         super().__init__()
#         self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
#         self.proj = nn.Linear(head_size * num_heads, N_EMBD)
#         self.dropout = nn.Dropout(DROPOUT)

#     def forward(self, x):
#         out = torch.cat([h(x) for h in self.heads], dim=-1)
#         out = self.dropout(self.proj(out))
#         return out

class FeedFoward(nn.Module):   #yeh MLP hai karpathy wala -> Feed forward hai sebastian wala
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * N_EMBD, N_EMBD)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    """ a simple linear layer followed by a non-linearity """

    # def __init__(self, n_embd):
    #     super().__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(N_EMBD, 4 * N_EMBD),
    #         nn.ReLU(),
    #         nn.Linear(4 * N_EMBD, N_EMBD),
    #         nn.Dropout(DROPOUT),
    #     )

    # def forward(self, x):
    #     return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = N_EMBD // n_head
        self.sa = CausalSelfAttention()
        self.ffwd = FeedFoward()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        print("This is vocab size:", vocab_size)
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(
            *[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)]
        )
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

        self.token_embedding_table.weight = self.lm_head.weight

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
                std = 0.02
                if hasattr(module, 'NANOGPT_SCALE_INIT'):
                    std *= (2 * N_LAYER) ** -0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= BLOCK_SIZE, f"Cannot forward sequence of length {T}, block size is only {BLOCK_SIZE}"


        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(0, T, dtype=torch.long, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            # B, T, C = logits.shape
            # logits = logits.view(B*T, C)
            # targets = targets.view(B*T)
            # loss = nn.functional.cross_entropy(logits, targets)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    # def save(self, path=MODEL_PATH):
    #     torch.save(self.state_dict(), path)

    # def load(self, path=MODEL_PATH):
    #     self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

    def save(self, path=MODEL_PATH):
        torch.save(self.state_dict(), path)

    # def load(self, path=MODEL_PATH):
    #     self.load_state_dict(torch.load(path))

    def load(self, path=MODEL_PATH):
    # Load the state dict
        state_dict = torch.load(path)

        # Rename the keys to match the expected ones (remove "orig_mod." prefix)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '')  # Remove 'orig_mod.' prefix
            new_state_dict[new_key] = value

        # Load the renamed state dict into the model
        self.load_state_dict(new_state_dict)


    def configure_optimizers(self, weight_decay=0.1, learning_rate=LEARNING_RATE, device=DEVICE):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_parameters = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_parameters = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": nodecay_parameters, "weight_decay": 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused = use_fused)
        return optimizer