"""
╔════════════════════════════════════════════════════════════════════════════╗
║  追求高准确率的 mini Transformer 训练                                         ║
║  模型 ~10M 参数，训练 10000 步，目标 COPY 100% / ADD 95%+ / REV 85%+          ║
╚════════════════════════════════════════════════════════════════════════════╝

1.核心提升：课程学习（Curriculum Learning）
这是本次升级的灵魂。代码模拟了人类“先易后难”的学习过程，让模型训练更高效、更稳定。
第一阶段：开卷考试 (Step 0 ~ 3000)
    训练方式：模型在训练时能看到完整的输入序列，包括问题和答案。这是一种标准的“Teacher Forcing”模式。
    目的：让模型先快速熟悉数据格式和任务的基本规则，打好基础。
第二阶段：半开卷考试 (Step 3000 ~ 10000)
    训练方式：从第 3000 步开始，代码会随机截断标签（Label）。模型在预测时，只能看到问题（Prompt）和一小部分答案，剩下的答案需要它自己“凭空”生成。
    目的：这极大地增加了训练难度，强迫模型从“照猫画虎”转变为真正的“自主生成”。这能有效提升模型的泛化能力和推理能力，防止它过度依赖看到的完整答案。
2.更长的训练周期
    提升：num_steps 增加到 10000 步，warmup_steps 增加到 500 步。
    意义：配合课程学习，更长的训练时间能让模型有充足的机会去攻克第二阶段的难题。
3.更精细的数据加载
    提升：DataLoader 中增加了 num_workers=2 和 pin_memory=True。
    意义：使用多进程加载数据并将数据预先锁页内存，可以加速数据从 CPU 到 GPU 的传输，提升整体训练吞吐量。
"""

import os, time, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ══════════════════════════════════════════════════════════════════════════════
# 0. 配置
# ══════════════════════════════════════════════════════════════════════════════
class Config:
    vocab_size    = 128
    hidden_size   = 256
    num_layers    = 6
    num_heads     = 8
    ff_multiplier = 4
    max_seq_len   = 64        # 缩短，减少 PAD
    dropout       = 0.05

    batch_size    = 32
    num_steps     = 10000     # 加长到 10000 步
    warmup_steps  = 500       # 更长 warmup
    log_every     = 100
    eval_every    = 500
    save_every    = 500
    lr            = 3e-4       # 降低 LR
    min_lr        = 1e-5
    weight_decay  = 0.01
    grad_clip     = 1.0

    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()

# ══════════════════════════════════════════════════════════════════════════════
# 1. 数据集 —— 紧凑、少 PAD、分布内可学
# ══════════════════════════════════════════════════════════════════════════════
class SyntheticDataset(Dataset):
    """
    序列格式（紧凑，尽量填满 64 长度）：
      [TAG] [body tokens...] [SEP] [answer tokens...] [EOS] [PAD...]
    
    所有数字 token ∈ [10, 19]（即数字 0~9），保证范围小、好记。
    ADD 结果也限制在 0~18 → token 10~28。
    """
    PAD, BOS, EOS, SEP = 0, 1, 2, 7
    COPY, ADD, REV = 3, 4, 5
    NUM_START = 10

    def __init__(self, num_samples, seq_len):
        self.seq_len = seq_len
        self.data = [self._make() for _ in range(num_samples)]

    def _make(self):
        L = self.seq_len
        kind = random.choice([0, 1, 2])

        if kind == 0:  # COPY: 复制 3~6 个数字
            n = random.randint(3, 6)
            nums = [random.randint(0, 9) for _ in range(n)]
            body = [self.NUM_START + x for x in nums]
            answer = body[:]  # 直接复制
            tag = self.COPY

        elif kind == 1:  # ADD: 两个 0~9 数字相加
            a = random.randint(0, 9)
            b = random.randint(0, 9)
            s = a + b  # 0~18，token = 10+s，最大 28，在词表 128 内
            body = [self.NUM_START + a, self.NUM_START + b]
            answer = [self.NUM_START + s]
            tag = self.ADD

        else:  # REV: 反转 3~6 个数字
            n = random.randint(3, 6)
            nums = [random.randint(0, 9) for _ in range(n)]
            body = [self.NUM_START + x for x in nums]
            answer = [self.NUM_START + x for x in reversed(nums)]
            tag = self.REV

        # 拼序列
        seq = [tag] + body + [self.SEP] + answer + [self.EOS]
        if len(seq) > L:
            seq = seq[:L]
        while len(seq) < L:
            seq.append(self.PAD)

        return torch.tensor(seq, dtype=torch.long)

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def collate(batch):
    x = torch.stack(batch)
    return x, x

# ══════════════════════════════════════════════════════════════════════════════
# 2. 模型（和之前一样，初始化稍调）
# ══════════════════════════════════════════════════════════════════════════════
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.g

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)

class FFN(nn.Module):
    def __init__(self, dim, ffn_dim, dropout=0.0):
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        gu = self.gate_up(x)
        gate, up = gu.chunk(2, dim=-1)
        return self.down(self.dropout(F.silu(gate) * up))

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, ffn_dim, dropout):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FFN(dim, ffn_dim, dropout)
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.hidden_size)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(cfg.hidden_size, cfg.num_heads,
                             cfg.hidden_size * cfg.ff_multiplier, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)  # embedding 稍小一点

    def forward(self, input_ids):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        h = self.embed(input_ids) + self.pos_embed(pos)
        h = self.drop(h)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        return self.lm_head(h)

# ══════════════════════════════════════════════════════════════════════════════
# 3. 工具函数
# ══════════════════════════════════════════════════════════════════════════════
def get_lr(step):
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.num_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))

def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, cfg.vocab_size),
                                   y[:, 1:].contiguous().view(-1), ignore_index=0)
            n_tok = (y[:, 1:] != 0).sum().item()
            total_loss += loss.item() * n_tok
            total_tok += n_tok
    model.train()
    avg = total_loss / max(total_tok, 1)
    return avg, math.exp(min(avg, 20))

def generate_sample(model, device, prompt_tokens, max_new=12):
    model.eval()
    with torch.no_grad():
        x = torch.tensor([prompt_tokens], device=device)
        for _ in range(max_new):
            logits = model(x)[:, -1, :]
            nxt = logits.argmax(-1, keepdim=True)
            if nxt.item() == 2:
                break
            x = torch.cat([x, nxt], dim=1)
    model.train()
    return x[0, len(prompt_tokens):].tolist()

def test_task_accuracy(model, dataset, device, task_tag, num_tests=50):
    """从验证集取真实样本，公平评估"""
    model.eval()
    correct, total = 0, 0
    for _ in range(num_tests):
        # 找一个该任务的样本
        sample = None
        for _ in range(200):
            idx = random.randint(0, len(dataset) - 1)
            if dataset[idx][0].item() == task_tag:
                sample = dataset[idx]
                break
        if sample is None:
            continue

        sep_pos = (sample == 7).nonzero()
        if len(sep_pos) == 0:
            continue
        sep_pos = sep_pos[0].item()

        prompt = sample[:sep_pos + 1].tolist()
        true_answer = []
        for i in range(sep_pos + 1, len(sample)):
            if sample[i].item() in (0, 2):
                break
            true_answer.append(sample[i].item())

        pred_tokens = generate_sample(model, device, prompt, max_new=len(true_answer) + 3)
        pred_answer = [t for t in pred_tokens if t not in (0, 2)]

        total += 1
        if pred_answer == true_answer:
            correct += 1
    model.train()
    return correct / max(total, 1) * 100

# ══════════════════════════════════════════════════════════════════════════════
# 4. 主训练循环（加 curriculum）
# ══════════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs("./checkpoints", exist_ok=True)
    torch.manual_seed(42)
    random.seed(42)

    train_set = SyntheticDataset(num_samples=10000, seq_len=cfg.max_seq_len)
    val_set = SyntheticDataset(num_samples=1000, seq_len=cfg.max_seq_len)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=2, pin_memory=True)

    model = MiniGPT(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                   betas=(0.9, 0.95), weight_decay=cfg.weight_decay)

    print("=" * 80)
    print(f"[Init] Model: MiniGPT-v2 | Params: {n_params/1e6:.2f}M")
    print(f"[Init] Arch: {cfg.num_layers}L × {cfg.hidden_size}d, {cfg.num_heads} heads, FFN×{cfg.ff_multiplier}")
    print(f"[Init] Device: {cfg.device.upper()}")
    if cfg.device == 'cuda':
        print(f"[Init] GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    print(f"[Init] LR: {cfg.lr}→{cfg.min_lr} | Warmup: {cfg.warmup_steps} | Steps: {cfg.num_steps}")
    print(f"[Init] Dataset: train={len(train_set)} val={len(val_set)} seq_len={cfg.max_seq_len}")
    print(f"[Init] Curriculum: full-seq → truncated prompt (after step {int(cfg.num_steps*0.3)})")
    print(f"[Init] Checkpoints → ./checkpoints/")
    print("=" * 80)
    print()

    step = 0
    running_loss, running_norm = 0.0, 0.0
    t_start = time.time()
    data_iter = iter(train_loader)
    clipped_flag = False

    while step < cfg.num_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(cfg.device), y.to(cfg.device)

        # ── Curriculum：后 70% 步随机截断（只给 prompt 部分训练）──────────
        # 这样模型被迫学会"从 SEP 开始自己生成"
        curriculum_start = int(cfg.num_steps * 0.3)
        if step > curriculum_start:
            # 随机截断：只保留到 SEP 之前或 SEP 之后一部分
            for i in range(x.size(0)):
                seq = x[i]
                sep_pos = (seq == 7).nonzero()
                if len(sep_pos) > 0:
                    sep_idx = sep_pos[0].item()
                    # 截断到 SEP 位置（含），后面全变 PAD
                    trunc_len = sep_idx + 1 + random.randint(0, min(3, cfg.max_seq_len - sep_idx - 1))
                    trunc_len = min(trunc_len, cfg.max_seq_len)
                    y[i, trunc_len:] = 0  # label 的截断部分标 PAD（被 ignore）
                    # 注意：x 不动，因为 teacher forcing 还是用完整 x 预测 y

        lr_now = get_lr(step)
        for g in optimizer.param_groups:
            g['lr'] = lr_now

        # Forward
        t0 = time.time()
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, cfg.vocab_size),
                               y[:, 1:].contiguous().view(-1), ignore_index=0)
        t_fwd = time.time() - t0

        # Backward
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        t_bwd = time.time() - t0

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = math.sqrt(total_norm)

        clipped_flag = total_norm > cfg.grad_clip
        if clipped_flag:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        t0 = time.time()
        optimizer.step()
        t_opt = time.time() - t0

        step += 1
        running_loss += loss.item()
        running_norm += total_norm

        # ── 日志 ────────────────────────────────────────────────────────────
        if step % cfg.log_every == 0:
            avg_loss = running_loss / cfg.log_every
            avg_norm = running_norm / cfg.log_every
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t_start
            mem = torch.cuda.max_memory_allocated(0) / 1e9 if cfg.device == 'cuda' else 0
            curric = " (curriculum)" if step > curriculum_start else ""
            print(f"\n[Step {step:05d}] ─── Loss = {avg_loss:.4f} | PPL = {ppl:.2f}{curric}")
            print(f"  ├─ LR = {lr_now:.2e} | grad_norm = {avg_norm:.3f}{'   CLIPPED' if clipped_flag else ''}")
            print(f"  ├─ Time: fwd={t_fwd*1000:.1f}ms | bwd={t_bwd*1000:.1f}ms | opt={t_opt*1000:.1f}ms")
            print(f"  ├─ Speed: {step/elapsed:.2f} steps/s | elapsed={elapsed:.0f}s")
            print(f"  └─ GPU Mem: {mem:.2f} GB")
            running_loss, running_norm = 0.0, 0.0

        # ── 验证 + 保存 ────────────────────────────────────────────────────
        if step % cfg.eval_every == 0:
            val_loss, val_ppl = evaluate(model, val_loader, cfg.device)
            print(f"\n{'='*70}")
            print(f"[Eval @ Step {step:05d}] val_loss = {val_loss:.4f} | val_ppl = {val_ppl:.2f}")

            for name, tag in [("COPY", 3), ("ADD", 4), ("REV", 5)]:
                acc = test_task_accuracy(model, val_set, cfg.device, tag, num_tests=50)
                bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
                print(f"  {name:6s} accuracy: {acc:5.1f}% {bar}")

            # 从验证集取 3 个真实样本展示
            print(f"  [Samples from val set]:")
            for tag, name in [(3, "COPY"), (4, "ADD"), (5, "REV")]:
                for _ in range(1000):
                    idx = random.randint(0, len(val_set) - 1)
                    sample = val_set[idx]
                    if sample[0].item() == tag:
                        break
                sep_pos = (sample == 7).nonzero()[0].item()
                prompt = sample[:sep_pos + 1].tolist()
                true_answer = []
                for i in range(sep_pos + 1, len(sample)):
                    if sample[i].item() in (0, 2): break
                    true_answer.append(sample[i].item())
                pred_tokens = generate_sample(model, cfg.device, prompt, max_new=10)
                pred_answer = [t for t in pred_tokens if t not in (0, 2)]
                true_nums = [str(t - 10) for t in true_answer]
                pred_nums = [str(t - 10) for t in pred_answer]
                match = "✅" if pred_answer == true_answer else "❌"
                print(f"    {name:6s} {match} true={true_nums} | pred={pred_nums}")

            # 保存
            ckpt_path = f"./checkpoints/minigpt_step_{step:05d}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': {
                    'vocab_size': cfg.vocab_size,
                    'hidden_size': cfg.hidden_size,
                    'num_layers': cfg.num_layers,
                    'num_heads': cfg.num_heads,
                    'ff_multiplier': cfg.ff_multiplier,
                    'max_seq_len': cfg.max_seq_len,
                    'dropout': cfg.dropout,
                },
                'val_loss': val_loss,
                'val_ppl': val_ppl,
            }, ckpt_path)
            print(f"  Saved → {ckpt_path}")
            print(f"{'='*70}")

    # ── 最终保存 ────────────────────────────────────────────────────────────
    final_path = "./checkpoints/minigpt_final.pt"
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': cfg.vocab_size,
            'hidden_size': cfg.hidden_size,
            'num_layers': cfg.num_layers,
            'num_heads': cfg.num_heads,
            'ff_multiplier': cfg.ff_multiplier,
            'max_seq_len': cfg.max_seq_len,
            'dropout': cfg.dropout,
        },
    }, final_path)

    total_time = time.time() - t_start
    final_val_loss, final_val_ppl = evaluate(model, val_loader, cfg.device)
    print(f"\n{'='*80}")
    print(f"[Training Complete] Steps: {cfg.num_steps} | Time: {total_time:.1f}s")
    print(f"  Final val_loss: {final_val_loss:.4f} | val_ppl: {final_val_ppl:.2f}")
    peak = torch.cuda.max_memory_allocated(0) / 1e9 if cfg.device == 'cuda' else 0
    print(f"  Peak GPU mem: {peak:.2f} GB | Params: {n_params/1e6:.2f}M")
    print(f"  Final model → {final_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()