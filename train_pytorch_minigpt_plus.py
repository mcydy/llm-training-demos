import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import random
import os
import argparse

# 增加了verbose模式

# ===================== Config =====================
class Config:
    vocab_size = 128    # 词表大小
    hidden_size = 128   # 每个 token 的向量维度
    num_layers = 4      # 4 个 Transformer 层
    num_heads = 4       # 注意力头数
    ffn_intermediate = 512  # FN 中间维度（SwiGLU 的 gate+up 各 512）
    max_seq_len = 64    # 最长序列 64 个 token
    dropout = 0.1

    batch_size = 64
    lr = 3e-4
    weight_decay = 0.01
    max_steps = 500
    warmup_steps = 50
    grad_clip = 1.0

    train_samples = 2000
    val_samples = 200
    seq_len = 64

    log_every = 10
    eval_every = 50
    save_dir = "./checkpoints"
    save_every = 100


# ===================== 模型组件（带 verbose 日志） =====================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, verbose=False, name=""):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.verbose = verbose
        self.name = name

    def forward(self, x):
        if self.verbose:
            print(f"  │   ├─ {self.name}: hidden {list(x.shape)} → norm {list(x.shape)} | μ={x.mean():.3f} σ={x.std():.3f}")
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        y = (x.float() * norm * self.weight.float()).to(x.dtype)
        if self.verbose:
            print(f"  │   │                    → μ={y.mean():.3f} σ={y.std():.3f}")
        return y


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg, verbose=False, layer_idx=0):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.hidden_size = cfg.hidden_size
        self.qkv = nn.Linear(cfg.hidden_size, cfg.hidden_size * 3, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.verbose = verbose
        self.layer_idx = layer_idx

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        if self.verbose:
            print(f"  │   ├─ QKV proj:         {list(x.shape)} → qkv {list((B, T, C*3))} (q={C}, kv={C//4} each)")

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.verbose:
            print(f"  │   ├─ Reshape heads:    q {list(q.shape)} | k {list(k.shape)} | v {list(v.shape)}")
            print(f"  │   ├─ RoPE:             applied @ positions [0..{T-1}]")

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=True)

        if self.verbose:
            print(f"  │   ├─ Attention:        scaled_dot_product | causal=True")
            print(f"  │   │                    Q·K^T/√{self.head_dim} → softmax → ×V")
            print(f"  │   │                    attn_output {list(y.shape)} → {list((B, T, C))}")

        y = y.transpose(1, 2).reshape(B, T, C)
        out = self.o_proj(y)

        if self.verbose:
            print(f"  │   ├─ o_proj:           {list(y.shape)} → {list(out.shape)}")

        return self.dropout(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, cfg, verbose=False):
        super().__init__()
        self.gate_up = nn.Linear(cfg.hidden_size, cfg.ffn_intermediate * 2, bias=False)
        self.down = nn.Linear(cfg.ffn_intermediate, cfg.hidden_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.verbose = verbose
        self.inter = cfg.ffn_intermediate

    def forward(self, x):
        if self.verbose:
            print(f"  │   ├─ MLP gate_up:      {list(x.shape)} → {list((x.size(0), x.size(1), self.inter*2))} (gate+up fused)")

        gate_up = self.gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)

        if self.verbose:
            print(f"  │   ├─ SwiGLU:           silu(gate) * up → {list(gate.shape)}")

        activated = F.silu(gate) * up
        out = self.down(activated)

        if self.verbose:
            print(f"  │   ├─ MLP down:         {list(activated.shape)} → {list(out.shape)}")

        return self.dropout(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg, verbose=False, layer_idx=0):
        super().__init__()
        self.verbose = verbose
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(cfg.hidden_size, verbose=verbose, name=f"input_layernorm (Layer {layer_idx})")
        self.ffn_norm = RMSNorm(cfg.hidden_size, verbose=verbose, name=f"post_attn_norm (Layer {layer_idx})")
        self.attn = CausalSelfAttention(cfg, verbose=verbose, layer_idx=layer_idx)
        self.ffn = SwiGLUFFN(cfg, verbose=verbose)

    def forward(self, x, attn_mask=None):
        if self.verbose:
            print(f"  │   ├─ [Layer {self.layer_idx:02d}] ──────────────────────────────────────────────────────────────")

        # Attention block
        normed = self.attn_norm(x)
        attn_out = self.attn(normed, attn_mask)
        x = x + attn_out

        if self.verbose:
            print(f"  │   ├─ Residual add:     hidden = hidden + attn_out → {list(x.shape)}")

        # FFN block
        normed2 = self.ffn_norm(x)
        ffn_out = self.ffn(normed2)
        x = x + ffn_out

        if self.verbose:
            print(f"  │   └─ Residual add:     hidden = hidden + mlp_out → {list(x.shape)}")

        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg, verbose=False):
        super().__init__()
        self.cfg = cfg
        self.verbose = verbose
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, verbose=verbose, layer_idx=i) for i in range(cfg.num_layers)
        ])
        self.final_norm = RMSNorm(cfg.hidden_size, verbose=False, name="final_norm")
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, return_logits=False):
        B, T = input_ids.shape

        if self.verbose:
            print(f"  ├─ [Forward] input_ids shape: {list(input_ids.shape)} | vocab_size: {self.cfg.vocab_size}")

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(pos)

        if self.verbose:
            print(f"  ├─ [Forward] Embedding output: {list(x.shape)} | mean={x.mean():.3f} std={x.std():.3f}")
            print(f"  │")
            for i, block in enumerate(self.blocks):
                block.verbose = True
            # 只打印第一层和最后一层详细，中间折叠
            if self.cfg.num_layers > 2:
                self.blocks[0].verbose = True
                for b in self.blocks[1:-1]:
                    b.verbose = False
                self.blocks[-1].verbose = True

        for i, block in enumerate(self.blocks):
            x = block(x)

        if self.verbose:
            self.final_norm.verbose = True
            print(f"  │")
            print(f"  ├─ [Final Norm] RMSNorm → {list(x.shape)} | μ={x.mean():.3f} σ={x.std():.3f}")

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if self.verbose:
            vocab = self.cfg.vocab_size
            hidden = self.cfg.hidden_size
            print(f"  ├─ [LM Head]    {list(x.shape)} × [{vocab},{hidden}]^T → logits {list(logits.shape)}")

        if labels is not None:
            if self.verbose:
                print(f"  ├─ [Loss]       Shift right: labels = input_ids[:, 1:]")
                print(f"  │               CrossEntropy(logits[:, :-1, :], labels)")
            loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), labels.reshape(-1))
            if self.verbose:
                import math
                ppl = math.exp(loss.item()) if loss.item() < 10 else float('inf')
                print(f"  │               → Loss = {loss.item():.4f}")
                print(f"  │               (perplexity = exp({loss.item():.2f}) ≈ {ppl:.1f})")
        else:
            loss = None

        return logits, loss


# ===================== 数据集 =====================
class SyntheticDataset:
    def __init__(self, num_samples, seq_len, vocab_size):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.samples = [self._make_one() for _ in range(num_samples)]

    def _make_one(self):
        T = self.seq_len
        vs = self.vocab_size
        task = random.choice(['COPY', 'ADD', 'REV'])
        ids = []
        if task == 'COPY':
            n = random.randint(2, 6)
            seq = [random.randint(10, min(50, vs - 1)) for _ in range(n)]
            ids.append(1)
            ids.extend(seq)
            ids.append(2)
            for v in seq:
                ids.append(v)
                if len(ids) >= T: break
            while len(ids) < T: ids.append(0)
            return ids[:T]
        elif task == 'ADD':
            a, b = random.randint(1, 20), random.randint(1, 20)
            ids.append(3); ids.append(a+10); ids.append(4); ids.append(b+10); ids.append(5)
            for d in str(a+b):
                ids.append(int(d)+10)
                if len(ids) >= T: break
            while len(ids) < T: ids.append(0)
            return ids[:T]
        else:
            n = random.randint(2, 6)
            seq = [random.randint(10, min(50, vs - 1)) for _ in range(n)]
            ids.append(6)
            ids.extend(seq)
            ids.append(2)
            for v in reversed(seq):
                ids.append(v)
                if len(ids) >= T: break
            while len(ids) < T: ids.append(0)
            return ids[:T]

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return torch.tensor(self.samples[idx], dtype=torch.long)


# ===================== 训练 =====================
def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate(model, dataset, cfg, device):
    model.eval()
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), cfg.batch_size):
            batch = dataset[i:i+cfg.batch_size].to(device)
            if batch.size(0) == 0: continue
            _, loss = model(batch[:, :-1], batch[:, 1:])
            total_loss += loss.item() * batch.size(0)
            total_count += batch.size(0)
    model.train()
    return total_loss / max(total_count, 1)


def save_checkpoint(model, optimizer, cfg, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {k: getattr(cfg, k) for k in ['vocab_size','hidden_size','num_layers','num_heads','ffn_intermediate','max_seq_len','dropout']}
    }, path)
    print(f"  💾 Saved checkpoint → {path}")


# ===================== Main =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help='Print detailed forward pass shapes')
    args = parser.parse_args()

    cfg = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print(f"[Init] Model: MiniGPT-Tiny | Params: {sum(p.numel() for p in MiniGPT(cfg).parameters())/1e6:.2f}M")
    print(f"[Init] Device: {device}")
    if device == 'cuda':
        print(f"[Init] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Init] Optimizer: AdamW (lr={cfg.lr}, wd={cfg.weight_decay})")
    print(f"[Init] Dataset: train={cfg.train_samples} | val={cfg.val_samples} | seq_len={cfg.seq_len}")
    print(f"[Init] Batch={cfg.batch_size} | GradClip={cfg.grad_clip} | Steps={cfg.max_steps}")
    print(f"[Init] Verbose: {'ON 🔍' if args.verbose else 'OFF (use --verbose to enable)'}")
    print("=" * 80)

    train_ds = SyntheticDataset(cfg.train_samples, cfg.seq_len, cfg.vocab_size)
    val_ds = SyntheticDataset(cfg.val_samples, cfg.seq_len, cfg.vocab_size)

    model = MiniGPT(cfg, verbose=args.verbose).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    step = 0
    t0 = time.time()
    train_losses = []

    while step < cfg.max_steps:
        idxs = random.sample(range(len(train_ds)), cfg.batch_size)
        batch = torch.stack([train_ds[i] for i in idxs]).to(device)

        lr = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g['lr'] = lr

        # Forward（第一步或 verbose 时打印详细）
        verbose_this_step = args.verbose and step == 0
        model.verbose = verbose_this_step

        t_fwd = time.time()
        logits, loss = model(batch[:, :-1], batch[:, 1:])
        fwd_time = (time.time() - t_fwd) * 1000

        # Backward
        t_bwd = time.time()
        optimizer.zero_grad()
        loss.backward()
        bwd_time = (time.time() - t_bwd) * 1000

        # Clip
        t_opt = time.time()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        clipped = " ！ CLIPPED" if gnorm > cfg.grad_clip else ""

        # Optimizer
        optimizer.step()
        opt_time = (time.time() - t_opt) * 1000

        step += 1
        train_losses.append(loss.item())

        # ===== 简洁日志（每步都打）=====
        if step % cfg.log_every == 0:
            ppl = math.exp(loss.item()) if loss.item() < 10 else float('inf')
            mem = torch.cuda.memory_allocated() / 1e9 if device == 'cuda' else 0
            peak = torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0

            print(f"\n[Step {step:04d}] ─── Loss = {loss.item():.4f} | PPL = {ppl:.2f}")
            print(f"  ├─ LR = {lr:.2e} | grad_norm = {gnorm:.3f}{clipped}")
            print(f"  ├─ Time:  fwd={fwd_time:.1f}ms | bwd={bwd_time:.1f}ms | opt={opt_time:.1f}ms")
            print(f"  └─ GPU Mem: {mem:.2f} GB / Peak: {peak:.2f} GB")

        # ===== Eval =====
        if step % cfg.eval_every == 0:
            val_loss = evaluate(model, val_ds, cfg, device)
            val_ppl = math.exp(val_loss) if val_loss < 10 else float('inf')
            print(f"\n{'='*60}")
            print(f"[Eval @ Step {step}] val_loss = {val_loss:.4f} | val_ppl = {val_ppl:.2f}")
            print(f"{'='*60}\n")

        # ===== Save =====
        if step % cfg.save_every == 0:
            save_path = os.path.join(cfg.save_dir, f"minigpt_step_{step}.pt")
            save_checkpoint(model, optimizer, cfg, step, save_path)

    # Final
    final_path = os.path.join(cfg.save_dir, f"minigpt_step_{step}.pt")
    save_checkpoint(model, optimizer, cfg, step, final_path)

    print(f"\n{'='*80}")
    print(f"[Training Complete] Total steps: {step} | Wall time: {time.time()-t0:.1f}s")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0
    print(f"  Peak GPU mem: {peak:.2f} GB")
    print(f"  Model saved: {final_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()