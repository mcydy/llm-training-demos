import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import random
import os

# ===================== Config =====================
class Config:
    # 模型
    vocab_size = 128
    hidden_size = 128
    num_layers = 4
    num_heads = 4
    ffn_intermediate = 512
    max_seq_len = 64
    dropout = 0.1

    # 训练
    batch_size = 64
    lr = 3e-4
    weight_decay = 0.01
    max_steps = 500
    warmup_steps = 50
    grad_clip = 1.0

    # 数据
    train_samples = 2000
    val_samples = 200
    seq_len = 64

    # 日志
    log_every = 10
    eval_every = 50

    # 保存
    save_dir = "./checkpoints"
    save_every = 100          # 每 100 步存一次
    resume_from = None        # 改成 "./checkpoints/minigpt_step_500.pt" 即可恢复训练

# ===================== 模型组件 =====================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm * self.weight.float()).to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.qkv = nn.Linear(cfg.hidden_size, cfg.hidden_size * 3, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)

        return self.dropout(self.o_proj(y))


class SwiGLUFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_up = nn.Linear(cfg.hidden_size, cfg.ffn_intermediate * 2, bias=False)
        self.down = nn.Linear(cfg.ffn_intermediate, cfg.hidden_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        gate_up = self.gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.dropout(self.down(F.silu(gate) * up))


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size)
        self.ffn_norm = RMSNorm(cfg.hidden_size)
        self.attn = CausalSelfAttention(cfg)
        self.ffn = SwiGLUFFN(cfg)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.attn_norm(x), attn_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(pos)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                labels.reshape(-1)
            )

        return logits, loss


# ===================== 合成数据集 =====================
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
                if len(ids) >= T:
                    break
            while len(ids) < T:
                ids.append(0)
            return ids[:T]

        elif task == 'ADD':
            a = random.randint(1, 20)
            b = random.randint(1, 20)
            ids.append(3)
            ids.append(a + 10)
            ids.append(4)
            ids.append(b + 10)
            ids.append(5)
            for d in str(a + b):
                ids.append(int(d) + 10)
                if len(ids) >= T:
                    break
            while len(ids) < T:
                ids.append(0)
            return ids[:T]

        else:  # REV
            n = random.randint(2, 6)
            seq = [random.randint(10, min(50, vs - 1)) for _ in range(n)]
            ids.append(6)
            ids.extend(seq)
            ids.append(2)
            for v in reversed(seq):
                ids.append(v)
                if len(ids) >= T:
                    break
            while len(ids) < T:
                ids.append(0)
            return ids[:T]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.long)


# ===================== 工具函数 =====================
def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate(model, dataset, cfg, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for i in range(0, len(dataset), cfg.batch_size):
            batch = dataset[i:i + cfg.batch_size].to(device)
            if batch.size(0) == 0:
                continue
            logits, loss = model(batch[:, :-1], batch[:, 1:])
            total_loss += loss.item() * batch.size(0)
            total_count += batch.size(0)
    model.train()
    return total_loss / max(total_count, 1)


def generate_sample(model, dataset, device, task_name, task_token_id, max_new=8):
    model.eval()
    result = f"{task_name}: no sample found"
    with torch.no_grad():
        for _ in range(10):  # 多试几次找到对应任务
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset[idx]
            if sample[0].item() == task_token_id:
                ids = sample.unsqueeze(0).to(device)
                logits, _ = model(ids[:, :-1])
                pred = logits[0, -1].argmax().item()
                arrow_idxs = (ids[0] == 2).nonzero()
                if len(arrow_idxs) > 0:
                    ans_start = arrow_idxs[0].item() + 1
                    true_ans = ids[0, ans_start:ans_start + max_new].tolist()
                    result = f"{task_name} ... → true_start={true_ans[:3]} pred_last={pred}"
                else:
                    result = f"{task_name} ... → pred_last={pred}"
                break
    model.train()
    return result


def save_checkpoint(model, optimizer, cfg, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'vocab_size': cfg.vocab_size,
            'hidden_size': cfg.hidden_size,
            'num_layers': cfg.num_layers,
            'num_heads': cfg.num_heads,
            'ffn_intermediate': cfg.ffn_intermediate,
            'max_seq_len': cfg.max_seq_len,
            'dropout': cfg.dropout,
        }
    }, path)
    print(f"  💾 Saved checkpoint → {path}")


def load_checkpoint(model, optimizer, cfg, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print(f"  📂 Loaded checkpoint from {path} (step {ckpt['step']})")
    return ckpt['step']


# ===================== Main =====================
def main():
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
    print(f"[Init] Save dir: {cfg.save_dir} | Save every: {cfg.save_every} steps")
    if cfg.resume_from:
        print(f"[Init] Resume from: {cfg.resume_from}")
    print("=" * 80)

    # 数据
    train_ds = SyntheticDataset(cfg.train_samples, cfg.seq_len, cfg.vocab_size)
    val_ds = SyntheticDataset(cfg.val_samples, cfg.seq_len, cfg.vocab_size)

    # 模型
    model = MiniGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 恢复训练（可选）
    start_step = 0
    if cfg.resume_from and os.path.exists(cfg.resume_from):
        start_step = load_checkpoint(model, optimizer, cfg, cfg.resume_from, device)

    # 训练
    step = start_step
    t0 = time.time()
    train_losses = []

    while step < cfg.max_steps:
        # 随机采样一个 batch
        idxs = random.sample(range(len(train_ds)), cfg.batch_size)
        batch = torch.stack([train_ds[i] for i in idxs]).to(device)

        # LR
        lr = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g['lr'] = lr

        # Forward
        t_fwd = time.time()
        logits, loss = model(batch[:, :-1], batch[:, 1:])
        fwd_time = (time.time() - t_fwd) * 1000

        # Backward
        t_bwd = time.time()
        optimizer.zero_grad()
        loss.backward()
        bwd_time = (time.time() - t_bwd) * 1000

        # Gradient clip
        t_opt = time.time()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        clipped = " ！ CLIPPED" if gnorm > cfg.grad_clip else ""

        # Optimizer step
        optimizer.step()
        opt_time = (time.time() - t_opt) * 1000

        step += 1
        train_losses.append(loss.item())

        # Log
        if step % cfg.log_every == 0:
            elapsed = time.time() - t0
            ppl = math.exp(loss.item()) if loss.item() < 10 else float('inf')
            mem = torch.cuda.memory_allocated() / 1e9 if device == 'cuda' else 0
            peak = torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0

            print(f"\n[Step {step:04d}] ─── Loss = {loss.item():.4f} | PPL = {ppl:.2f}")
            print(f"  ├─ LR = {lr:.2e} | grad_norm = {gnorm:.3f}{clipped}")
            print(f"  ├─ Time:  fwd={fwd_time:.1f}ms | bwd={bwd_time:.1f}ms | opt={opt_time:.1f}ms")
            print(f"  └─ GPU Mem: {mem:.2f} GB / Peak: {peak:.2f} GB")

        # Eval
        if step % cfg.eval_every == 0:
            val_loss = evaluate(model, val_ds, cfg, device)
            val_ppl = math.exp(val_loss) if val_loss < 10 else float('inf')

            print(f"\n{'='*60}")
            print(f"[Eval @ Step {step}] val_loss = {val_loss:.4f} | val_ppl = {val_ppl:.2f}")
            print(f"[Gen]  samples:")
            print(f"       {generate_sample(model, val_ds, device, 'COPY', 1)}")
            print(f"       {generate_sample(model, val_ds, device, 'ADD', 3)}")
            print(f"       {generate_sample(model, val_ds, device, 'REV', 6)}")
            print(f"{'='*60}\n")

        # 定期保存
        if step % cfg.save_every == 0:
            save_path = os.path.join(cfg.save_dir, f"minigpt_step_{step}.pt")
            save_checkpoint(model, optimizer, cfg, step, save_path)

    # 最终保存
    final_path = os.path.join(cfg.save_dir, f"minigpt_step_{step}.pt")
    save_checkpoint(model, optimizer, cfg, step, final_path)

    # 最终
    print(f"\n{'='*80}")
    print(f"[Training Complete] Total steps: {step} | Wall time: {time.time()-t0:.1f}s")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0
    print(f"  Peak GPU mem: {peak:.2f} GB")
    print(f"  Model saved: {final_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()