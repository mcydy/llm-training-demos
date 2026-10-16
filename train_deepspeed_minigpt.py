#!/usr/bin/env python3
"""
支持 DeepSpeed 分布式训练的 Mini Transformer
在合成数据集上学习 COPY / ADD / REV 三种任务

运行方式:
  # 单卡
  deepspeed --num_gpus=1 train_deepspeed_minigpt.py

  # 多卡
  deepspeed --num_gpus=4 train_deepspeed_minigpt.py

  # 指定节点
  deepspeed --num_nodes=2 --num_gpus_per_node=2 train_deepspeed_minigpt.py
"""

import os
import sys
import math
import random
import time
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1. DeepSpeed 初始化（必须在其他 torch 分布式操作之前）
# ============================================================
import deepspeed
import deepspeed.comm as dist

# 初始化分布式环境
deepspeed.init_distributed()
local_rank = dist.get_rank()
world_size = dist.get_world_size()

# 设置当前进程的设备
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    device = torch.device("cpu")

# ============================================================
# 2. 配置
# ============================================================
class Config:
    # 模型
    vocab_size    = 128
    hidden_size   = 256
    num_layers    = 6
    num_heads     = 8
    ff_multiplier = 4
    max_seq_len   = 64
    dropout       = 0.05

    # 训练
    batch_size    = 32
    num_steps     = 10000
    warmup_steps  = 500
    log_every     = 100
    eval_every    = 500
    save_every    = 500
    lr            = 3e-4
    min_lr        = 1e-5
    weight_decay  = 0.01
    grad_clip     = 1.0

    # 数据
    train_samples = 10000
    val_samples   = 1000
    seq_len       = 64

    # 课程学习
    curriculum_start_ratio = 0.3   # 30% 步数后开启课程学习

    # 路径
    save_dir = "./checkpoints"
    resume_from = None

    # 打印
    verbose = False

cfg = Config()

# ============================================================
# 3. 合成数据集（COPY / ADD / REV）
# ============================================================
class SyntheticDataset(Dataset):
    """
    序列格式:
      [TAG] [body tokens...] [SEP] [answer tokens...] [EOS] [PAD...]

    标记符:
      PAD=0  BOS=1  EOS=2  SEP=7
      COPY=3  ADD=4  REV=5
    数字 token: 10~19 对应数字 0~9

    任务说明:
      COPY: 输入 n 个数字 -> 原样复制输出
      ADD:  输入两个数字 a, b -> 输出 a+b 的各位数字
      REV:  输入 n 个数字 -> 反向输出
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
            answer = body[:]
            tag = self.COPY

        elif kind == 1:  # ADD: 两个 0~9 数字相加
            a = random.randint(0, 9)
            b = random.randint(0, 9)
            s = a + b
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate(batch):
    x = torch.stack(batch)
    return x, x  # input = label (teacher forcing)


# ============================================================
# 4. 模型组件
# ============================================================
class RMSNorm(nn.Module):
    """RMS Norm - LLaMA 用的归一化，比 LayerNorm 更轻量"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.g


class CausalSelfAttention(nn.Module):
    """因果自注意力 - 带 causal mask，每个位置只能看到前面的 token"""
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
        # causal mask: 只保留下三角部分
        mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU 前馈网络 - 现代 LLM 标配，比传统 GELU FFN 效果更好"""
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
    """Transformer Block - Pre-Norm + 残差连接"""
    def __init__(self, dim, heads, ffn_dim, dropout):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniGPT(nn.Module):
    """
    MiniGPT - 完整语言模型
    包含: token embedding + 位置编码 + 多层 TransformerBlock + final RMSNorm + lm_head
    lm_head 权重与 token_embed 共享（weight tying）
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.hidden_size)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(
                cfg.hidden_size, cfg.num_heads,
                cfg.hidden_size * cfg.ff_multiplier, cfg.dropout
            )
            for _ in range(cfg.num_layers)
        ])
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        # weight tying: lm_head 和 token_embed 共享权重
        self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, input_ids):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        h = self.embed(input_ids) + self.pos_embed(pos)
        h = self.drop(h)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        return self.lm_head(h)


# ============================================================
# 5. 学习率调度器
# ============================================================
def get_lr(step):
    """Warmup + Cosine 衰减到 min_lr"""
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.num_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================
# 6. 评估函数
# ============================================================
def evaluate(model, loader, device):
    """在验证集上计算 cross-entropy loss 和 PPL"""
    model.eval()
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, cfg.vocab_size),
                y[:, 1:].contiguous().view(-1),
                ignore_index=0
            )
            n_tok = (y[:, 1:] != 0).sum().item()
            total_loss += loss.item() * n_tok
            total_tok += n_tok
    model.train()
    avg = total_loss / max(total_tok, 1)
    return avg, math.exp(min(avg, 20))


def generate_sample(model, device, prompt_tokens, max_new=12):
    """自回归生成: 从 prompt 开始逐个预测下一个 token"""
    model.eval()
    with torch.no_grad():
        x = torch.tensor([prompt_tokens], device=device)
        for _ in range(max_new):
            logits = model(x)[:, -1, :]
            nxt = logits.argmax(-1, keepdim=True)
            if nxt.item() == 2:  # EOS
                break
            x = torch.cat([x, nxt], dim=1)
    model.train()
    return x[0, len(prompt_tokens):].tolist()


def test_task_accuracy(model, dataset, device, task_tag, num_tests=50):
    """逐任务准确率评估: 从验证集取真实样本，公平评估"""
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

        # 找 SEP 位置，分离 prompt 和答案
        sep_pos = (sample == 7).nonzero()
        if len(sep_pos) == 0:
            continue
        sep_pos = sep_pos[0].item()

        prompt = sample[:sep_pos + 1].tolist()
        true_answer = []
        for i in range(sep_pos + 1, len(sample)):
            if sample[i].item() in (0, 2):  # PAD 或 EOS
                break
            true_answer.append(sample[i].item())

        pred_tokens = generate_sample(model, device, prompt, max_new=len(true_answer) + 3)
        pred_answer = [t for t in pred_tokens if t not in (0, 2)]

        total += 1
        if pred_answer == true_answer:
            correct += 1
    model.train()
    return correct / max(total, 1) * 100


# ============================================================
# 7. DeepSpeed 配置
# ============================================================
ds_config = {
    "train_micro_batch_size_per_gpu": cfg.batch_size,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "betas": [0.9, 0.95]
        }
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"},
        "allgather_partitions": True,
        "reduce_scatter": True,
        "overlap_comm": True,
        "contiguous_gradients": True
    },
    "steps_per_print": cfg.log_every,
    "wall_clock_breakdown": False,
}


# ============================================================
# 8. 主函数
# ============================================================
def main():
    # 设置随机种子（保证可复现）
    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    os.makedirs(cfg.save_dir, exist_ok=True)

    # 打印配置
    if local_rank == 0:
        print("=" * 80)
        print("MiniGPT-DeepSpeed - COPY / ADD / REV 任务训练")
        print("=" * 80)
        print(f"[Rank {local_rank}] World size: {world_size}")
        print(f"[Rank {local_rank}] Device: {device}")
        if torch.cuda.is_available():
            print(f"[Rank {local_rank}] GPU: {torch.cuda.get_device_name(local_rank)}")
            print(f"[Rank {local_rank}] VRAM: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB")
        print(f"[Config] vocab={cfg.vocab_size} hidden={cfg.hidden_size} layers={cfg.num_layers} heads={cfg.num_heads}")
        print(f"[Config] FFN multiplier={cfg.ff_multiplier} max_seq={cfg.max_seq_len} dropout={cfg.dropout}")
        print(f"[Config] batch={cfg.batch_size} steps={cfg.num_steps} warmup={cfg.warmup_steps}")
        print(f"[Config] LR: {cfg.lr} -> {cfg.min_lr} | min_lr={cfg.min_lr}")
        print(f"[Config] Curriculum start: step {int(cfg.num_steps * cfg.curriculum_start_ratio)}")
        print(f"[Config] Save dir: {cfg.save_dir}")
        print("=" * 80)

    # 数据
    train_set = SyntheticDataset(num_samples=cfg.train_samples, seq_len=cfg.seq_len)
    val_set = SyntheticDataset(num_samples=cfg.val_samples, seq_len=cfg.seq_len)

    # DataLoader
    # 注意: DeepSpeed 下需要使用 DistributedSampler 才能保证多卡数据不重复
    # 这里为了简化，单卡或小数据集直接用普通 DataLoader
    # 如需严格分布式，请替换为 torch.utils.data.distributed.DistributedSampler
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate, num_workers=2, pin_memory=True
    )

    # 模型
    model = MiniGPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    if local_rank == 0:
        print(f"\n[Model] Total params: {n_params / 1e6:.2f}M")
        print(f"[Model] Arch: {cfg.num_layers}L x {cfg.hidden_size}d, {cfg.num_heads} heads, SwiGLU FFN")
        print(f"[Model] Weight tying: Yes")

    # DeepSpeed 初始化（包装模型和优化器）
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    # 恢复训练（可选）
    start_step = 0
    if cfg.resume_from and os.path.exists(cfg.resume_from):
        if local_rank == 0:
            print(f"\n[Resume] Loading checkpoint from {cfg.resume_from}")
        model_engine.load_checkpoint(cfg.resume_from)
        # 从 checkpoint 文件名提取步数
        match = re.search(r'step_(\d+)', cfg.resume_from)
        start_step = int(match.group(1)) if match else 0

    # 训练循环
    step = start_step
    running_loss, running_norm = 0.0, 0.0
    t_start = time.time()
    data_iter = iter(train_loader)
    curriculum_start = int(cfg.num_steps * cfg.curriculum_start_ratio)

    if local_rank == 0:
        print(f"\n[Training] Starting from step {start_step}...")
        print(f"[Training] Curriculum learning enabled after step {curriculum_start}")
        print()

    while step < cfg.num_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)

        # --- 课程学习：后 70% 步随机截断标签 ---
        # 让模型先学会"照答案抄"(teacher forcing)，再学会"自己生成"
        if step > curriculum_start:
            for i in range(x.size(0)):
                seq = x[i]
                sep_pos = (seq == 7).nonzero()
                if len(sep_pos) > 0:
                    sep_idx = sep_pos[0].item()
                    # 截断到 SEP 位置之后 0~3 个 token
                    trunc_len = sep_idx + 1 + random.randint(0, min(3, cfg.max_seq_len - sep_idx - 1))
                    trunc_len = min(trunc_len, cfg.max_seq_len)
                    y[i, trunc_len:] = 0  # 截断部分标 PAD（被 ignore_index 忽略）

        # 学习率
        lr_now = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        # Forward
        t0 = time.time()
        logits = model_engine(x)
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, cfg.vocab_size),
            y[:, 1:].contiguous().view(-1),
            ignore_index=0
        )
        t_fwd = time.time() - t0

        # Backward
        t0 = time.time()
        model_engine.backward(loss)
        t_bwd = time.time() - t0

        # 梯度裁剪
        total_norm = 0.0
        if hasattr(model_engine, "module"):
            model_params = model_engine.module.parameters()
        else:
            model_params = model_engine.parameters()
        for p in model_params:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = math.sqrt(total_norm)

        clipped_flag = total_norm > cfg.grad_clip
        if clipped_flag:
            model_engine.clip_grad_norm_(cfg.grad_clip)

        # Optimizer step
        t0 = time.time()
        model_engine.step()
        t_opt = time.time() - t0

        step += 1
        running_loss += loss.item()
        running_norm += total_norm

        # --- 日志 ---
        if step % cfg.log_every == 0 and local_rank == 0:
            avg_loss = running_loss / cfg.log_every
            avg_norm = running_norm / cfg.log_every
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t_start
            curric = " (curriculum)" if step > curriculum_start else ""
            mem = torch.cuda.max_memory_allocated(local_rank) / 1e9 if torch.cuda.is_available() else 0

            print(f"\n[Step {step:05d}] --- Loss = {avg_loss:.4f} | PPL = {ppl:.2f}{curric}")
            print(f"  |-- LR = {lr_now:.2e} | grad_norm = {avg_norm:.3f}{'  WARNING CLIPPED' if clipped_flag else ''}")
            print(f"  |-- Time: fwd={t_fwd*1000:.1f}ms | bwd={t_bwd*1000:.1f}ms | opt={t_opt*1000:.1f}ms")
            print(f"  |-- Speed: {step / elapsed:.2f} steps/s | elapsed={elapsed:.0f}s")
            print(f"  '-- GPU Mem: {mem:.2f} GB")
            running_loss, running_norm = 0.0, 0.0

        # --- 验证 + 保存 ---
        if step % cfg.eval_every == 0:
            val_loss, val_ppl = evaluate(model_engine, val_loader, device)

            if local_rank == 0:
                print(f"\n{'=' * 70}")
                print(f"[Eval @ Step {step:05d}] val_loss = {val_loss:.4f} | val_ppl = {val_ppl:.2f}")

                # 逐任务准确率
                for name, tag in [("COPY", 3), ("ADD", 4), ("REV", 5)]:
                    acc = test_task_accuracy(model_engine, val_set, device, tag, num_tests=50)
                    bar = chr(0x2588) * int(acc / 5) + chr(0x2591) * (20 - int(acc / 5))
                    print(f"  {name:6s} accuracy: {acc:5.1f}% {bar}")

                # 真实样本生成展示
                print(f"  [Sample generation from val set]:")
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
                        if sample[i].item() in (0, 2):
                            break
                        true_answer.append(sample[i].item())
                    pred_tokens = generate_sample(model_engine, device, prompt, max_new=10)
                    pred_answer = [t for t in pred_tokens if t not in (0, 2)]
                    true_nums = [str(t - 10) for t in true_answer]
                    pred_nums = [str(t - 10) for t in pred_answer]
                    match = "OK" if pred_answer == true_answer else "MISS"
                    print(f"    {name:6s} {match} true={true_nums} | pred={pred_nums}")

                # 保存 checkpoint
                ckpt_path = os.path.join(cfg.save_dir, f"minigpt_ds_step_{step:05d}")
                model_engine.save_checkpoint(ckpt_path)
                print(f"  [Saved] -> {ckpt_path}")
                print(f"{'=' * 70}")

    # --- 最终保存 ---
    if local_rank == 0:
        final_path = os.path.join(cfg.save_dir, "minigpt_ds_final")
        model_engine.save_checkpoint(final_path)
        print(f"\n{'=' * 80}")
        print(f"[Training Complete] Total steps: {step} | Wall time: {time.time() - t_start:.1f}s")
        print(f"  Final model saved: {final_path}")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
