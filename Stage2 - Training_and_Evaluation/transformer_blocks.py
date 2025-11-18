import torch
import torch.nn as nn

## GPT CONFIGURATIONS ##
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias (W*x + b)
}

## MUTLIHEAD ATTENTION ##
class MultiheadAttention(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_heads = configs['n_heads']
        self.head_dim = int(configs['emb_dim'] / configs['n_heads'])
        self.Wq = nn.Linear(configs['emb_dim'], configs['emb_dim'])
        self.Wk = nn.Linear(configs['emb_dim'], configs['emb_dim'])
        self.Wv = nn.Linear(configs['emb_dim'], configs['emb_dim'])
        self.proj = nn.Linear(configs['emb_dim'], configs['emb_dim'])
        self.register_buffer( 'mask',torch.triu(torch.ones(configs['context_length'], configs['context_length']), diagonal=1))
        self.dropout = torch.nn.Dropout(configs['drop_rate'])

    def forward(self, x):
        # shape of x (B, T (context_len), D) 
        B, T, D = x.shape
        query = self.Wq(x)    # (B, T, D)
        key = self.Wk(x)
        value = self.Wv(x)

        # unrolling these weight Q/K/V metrices from (B, T, D) -> (B, T, n_heads, head_dim)
        query = query.view(B, T, self.n_heads, self.head_dim)
        key = key.view(B, T, self.n_heads, self.head_dim)
        value = value.view(B, T, self.n_heads, self.head_dim)

        # transpose (batch, context_len, n_heads, head_dim) -> (batch, n_heads, context_len, head_dim)
        keys = key.transpose(1, 2)
        queries = query.transpose(1, 2)
        values = value.transpose(1, 2)

        # time to calculate attention weights of shape (B, n_heads, context_len, context_len)
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill_(self.mask.bool()[:T, :T], -torch.inf)
        attn_scores = attn_scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        scaled_attn_weights = torch.softmax(attn_scores, dim=-1)
        scaled_attn_weights = self.dropout(scaled_attn_weights)
        outputs = scaled_attn_weights @ values

        # reformat context vectors 
        # (batch, heads, context_len, head_dim) -> (batch, context_len, heads, head_dim)
        outputs = outputs.transpose(1, 2)
        outputs = outputs.contiguous().view(B, T, self.n_heads * self.head_dim)

        context_vector = self.proj(outputs)
        return context_vector
    
## LAYER NORMALIZATION ## 
class LayerNorm(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(configs['emb_dim']))
        self.shift = nn.Parameter(torch.zeros(configs['emb_dim']))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    

## GeLU ACTIVATION ##
class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
## FEED FORWARD ##
class FeedForward(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(configs['emb_dim'], 4*configs['emb_dim']),
            GeLU(),
            nn.Linear(4*configs['emb_dim'], configs['emb_dim'])
        )

    def forward(self, x):
        return self.feedforward(x)
    
## TRANSFORMER BLOCK ##
import torch 
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.attn = MultiheadAttention(configs)
        self.ff = FeedForward(configs)
        self.norm1 = LayerNorm(configs)
        self.norm2 = LayerNorm(configs)
        self.drop_shortcut = nn.Dropout(configs["drop_rate"])

    def forward(self, x):
        # shape of x is (B, T, D)
        shortcut = x 
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x        
    
## GPT2 CLASS ##
import torch 
import torch.nn as nn

class GPTModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # token embeddings
        self.tok_emb = nn.Embedding(configs['vocab_size'], configs['emb_dim'])
        # positionl encodings 
        self.pos_emb = torch.nn.Embedding(configs['context_length'], configs['emb_dim'])
        # dropout
        self.drop_emb = nn.Dropout(configs['drop_rate'])

        # transformer block - encoder only in case of GPT
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(configs) for _ in range(configs['n_layers'])]
        )

        self.final_norm = LayerNorm(configs)
        self.out_head = nn.Linear(
            configs["emb_dim"], configs["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

        