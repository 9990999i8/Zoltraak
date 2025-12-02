import os
import math
import time
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F
import struct
import argparse
import sys

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

class Config:
    def __init__(self, mode='colab'):
        if mode == 'fast':
            # Fast mode for local verification (1 min run)
            self.block_size = 64
            self.vocab_size = 512 # Very small vocab for speed
            self.n_layer = 2
            self.n_head = 2
            self.n_embd = 64
            self.dropout = 0.0
            self.batch_size = 4
            self.max_iters = 10
            self.learning_rate = 1e-3
            self.eval_interval = 5
            self.eval_iters = 1
            self.device = 'cpu'
            self.data_subset_size = 10000 # Only use 10KB of data
        else:
            # Colab mode for real training (Target: 1-2 hours on T4)
            self.block_size = 256 # Context length
            self.vocab_size = 4096 # BPE Vocab size
            self.n_layer = 6
            self.n_head = 6
            self.n_embd = 384
            self.dropout = 0.2
            self.batch_size = 64
            self.max_iters = 5000 # Adjust this based on speed, 5000 iters on T4 should be decent
            self.learning_rate = 3e-4
            self.eval_interval = 500
            self.eval_iters = 200
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.data_subset_size = 0 # Use all data

# -----------------------------------------------------------------------------
# Tokenizer (Minimal BPE)
# -----------------------------------------------------------------------------

class BPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.merges = {} # (int, int) -> int
        self.vocab = {}  # int -> bytes
        self.special_tokens = {}

    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, text, verbose=True):
        # Initial vocab is just bytes
        ids = list(text.encode("utf-8"))
        original_len = len(ids)
        if verbose: print(f"Training BPE on {len(ids)} bytes...")

        # We start with 256 byte tokens
        num_merges = self.vocab_size - 256
        
        # Simple iterative BPE
        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
            if verbose and i % 100 == 0:
                print(f"Merge {i+1}/{num_merges}: {pair} -> {idx}")
        
        if verbose: print(f"Compression: {original_len} -> {len(ids)} ({original_len/len(ids):.2f}X)")
        
        # Build final vocab map
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = self.get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = self.merge(ids, pair, idx)
        return ids
    
    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")

    def save(self, filename):
        # Format:
        # [int32] vocab_size
        # Loop vocab_size:
        #   [int32] len
        #   [bytes] token_bytes
        # [int32] num_merges
        # Loop num_merges:
        #   [int32] p0, [int32] p1, [int32] new_idx
        
        print(f"Saving tokenizer to {filename}...")
        with open(filename, 'wb') as f:
            # Write Vocab
            f.write(struct.pack('>I', self.vocab_size))
            # Ensure we write all tokens up to vocab_size (some might not exist if training stopped early)
            for i in range(self.vocab_size):
                token_bytes = self.vocab.get(i, b'')
                f.write(struct.pack('>I', len(token_bytes)))
                f.write(token_bytes)
            
            # Write Merges
            sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
            f.write(struct.pack('>I', len(sorted_merges)))
            for (p0, p1), idx in sorted_merges:
                f.write(struct.pack('>III', p0, p1, idx))

# -----------------------------------------------------------------------------
# Model Architecture (GPT)
# -----------------------------------------------------------------------------

class Head(nn.Module):
    def __init__(self, head_size, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, Hs)
        q = self.query(x) # (B, T, Hs)
        
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, Hs) @ (B, Hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x) # (B, T, Hs)
        out = wei @ v # (B, T, T) @ (B, T, Hs) -> (B, T, Hs)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(config, head_size)
        self.ffwd = FeedFoward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd) # final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # focus only on the last time step
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    def export(self, filename):
        print(f"Exporting model to {filename}...")
        # Flat binary format:
        # Magic(4), d_model(4), d_ff(4), n_layer(4), n_head(4), vocab_size(4), max_seq_len(4)
        # Weights (float32)
        
        # d_ff is implicitly 4*d_model in our simple implementation, but we should store it if we want to be generic. 
        # But our Java implementation will just assume 4*d_model for now or I can add it to config.
        # Let's just write d_model and Java computes d_ff = 4*d_model.
        
        with open(filename, 'wb') as f:
            # Header
            f.write(struct.pack('>I', 0x54494E59)) # Magic TINY
            f.write(struct.pack('>I', self.config.n_embd))
            f.write(struct.pack('>I', self.config.n_embd * 4)) # d_ff
            f.write(struct.pack('>I', self.config.n_layer))
            f.write(struct.pack('>I', self.config.n_head))
            f.write(struct.pack('>I', self.config.vocab_size))
            f.write(struct.pack('>I', self.config.block_size))
            
            # Helper to write tensor
            def write_tensor(t):
                # Ensure float32 and cpu
                t_cpu = t.detach().cpu().to(torch.float32)
                f.write(t_cpu.numpy().tobytes())
            
            # Weights in specific order for Java loader
            # 1. Token Embeddings
            write_tensor(self.token_embedding_table.weight)
            # 2. Position Embeddings
            write_tensor(self.position_embedding_table.weight)
            
            # 3. Blocks
            for block in self.blocks:
                # Attention
                # We need to be careful with how we stored weights in MultiHeadAttention
                # In this impl, we have a ModuleList of Heads. Java will likely prefer one big matrix for Q, K, V.
                # But iterating heads is fine too if we write them sequentially.
                # Java strategy: Loop n_head: load Q, load K, load V.
                for head in block.sa.heads:
                    write_tensor(head.query.weight)
                for head in block.sa.heads:
                    write_tensor(head.key.weight)
                for head in block.sa.heads:
                    write_tensor(head.value.weight)
                
                # Proj
                write_tensor(block.sa.proj.weight)
                write_tensor(block.sa.proj.bias)
                
                # LN1
                write_tensor(block.ln1.weight)
                write_tensor(block.ln1.bias)
                
                # MLP
                # net[0] is Linear 1
                write_tensor(block.ffwd.net[0].weight)
                write_tensor(block.ffwd.net[0].bias)
                # net[2] is Linear 2
                write_tensor(block.ffwd.net[2].weight)
                write_tensor(block.ffwd.net[2].bias)
                
                # LN2
                write_tensor(block.ln2.weight)
                write_tensor(block.ln2.bias)
                
            # 4. Final LN
            write_tensor(self.ln_f.weight)
            write_tensor(self.ln_f.bias)
            
            # 5. LM Head
            write_tensor(self.lm_head.weight)
            write_tensor(self.lm_head.bias)
            
        print("Export complete.")

# -----------------------------------------------------------------------------
# Main Training Logic
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='colab', choices=['colab', 'fast'])

    # Check for Colab/Jupyter environment to handle kernel arguments gracefully
    if 'google.colab' in sys.modules or 'ipykernel' in sys.modules:
        print("Detected notebook environment. Ignoring kernel arguments.")
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
    
    config = Config(mode=args.mode)
    print(f"Running in {args.mode} mode on {config.device}")

    # 1. Download Data
    # For robust training we want more than just TinyShakespeare.
    # We will try to download a few books from Gutenberg if in colab mode, else just Shakespeare.
    
    data_urls = [
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    ]
    if args.mode == 'colab':
        # Add some more public domain text for bulk (Alice in Wonderland, etc)
        data_urls.append("https://www.gutenberg.org/files/11/11-0.txt") 
        data_urls.append("https://www.gutenberg.org/files/1342/1342-0.txt")

    text = ""
    for url in data_urls:
        try:
            print(f"Downloading {url}...")
            r = requests.get(url)
            text += r.text + "\n"
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    
    print(f"Total dataset size: {len(text)/1024/1024:.2f} MB")
    
    if config.data_subset_size > 0:
        text = text[:config.data_subset_size]
        print(f"Truncated to {len(text)} characters for fast mode.")

    # 2. Train Tokenizer
    tokenizer = BPETokenizer(config.vocab_size)
    tokenizer.train(text, verbose=True)
    tokenizer.save("tokenizer.bin")
    
    # 3. Prepare Data
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Training tokens: {len(train_data)}")
    
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
        x = torch.stack([data[i:i+config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
        x, y = x.to(config.device), y.to(config.device)
        return x, y

    # 4. Train Model
    model = GPT(config)
    m = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print(f"Model parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")

    for iter in range(config.max_iters):
        if iter % config.eval_interval == 0:
            losses = {}
            model.eval()
            for split in ['train', 'val']:
                loss_list = []
                for k in range(config.eval_iters):
                    X, Y = get_batch(split)
                    logits, loss = model(X, Y)
                    loss_list.append(loss.item())
                losses[split] = sum(loss_list) / len(loss_list)
            model.train()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # 5. Export
    model.export("model.bin")

if __name__ == '__main__':
    main()
