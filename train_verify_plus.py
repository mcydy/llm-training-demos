import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os
import sys

# ===================== Config（必须和训练时一样） =====================
class Config:
    vocab_size = 128
    hidden_size = 128
    num_layers = 4
    num_heads = 4
    ffn_intermediate = 512
    max_seq_len = 64
    seq_len = 64
    dropout = 0.0

# ===================== 模型定义（和训练时一模一样） =====================
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
    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=True)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, C))

class SwiGLUFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_up = nn.Linear(cfg.hidden_size, cfg.ffn_intermediate * 2, bias=False)
        self.down = nn.Linear(cfg.ffn_intermediate, cfg.hidden_size, bias=False)
    def forward(self, x):
        gate_up = self.gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)

class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size)
        self.ffn_norm = RMSNorm(cfg.hidden_size)
        self.attn = CausalSelfAttention(cfg)
        self.ffn = SwiGLUFFN(cfg)
    def forward(self, x, attn_mask=None):
        return x + self.ffn(self.ffn_norm(x + self.attn(self.attn_norm(x), attn_mask)))

class MiniGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)  # [B, T, vocab_size] logits

# ===================== 数据集（和训练时一样） =====================
class SyntheticDataset:
    def __init__(self, num_samples, seq_len, vocab_size):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.samples = [self._make_one() for _ in range(num_samples)]

    def _make_one(self):
        T, vs = self.seq_len, self.vocab_size
        task = random.choice(['COPY', 'ADD', 'REV'])
        ids = []
        if task == 'COPY':
            n = random.randint(2, 6)
            seq = [random.randint(10, min(50, vs - 1)) for _ in range(n)]
            ids = [1] + seq + [2] + seq
        elif task == 'ADD':
            a, b = random.randint(1, 20), random.randint(1, 20)
            ids = [3, a + 10, 4, b + 10, 5] + [int(d) + 10 for d in str(a + b)]
        else:
            n = random.randint(2, 6)
            seq = [random.randint(10, min(50, vs - 1)) for _ in range(n)]
            ids = [6] + seq + [2] + list(reversed(seq))
        while len(ids) < T:
            ids.append(0)
        return ids[:T]

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return torch.tensor(self.samples[idx], dtype=torch.long)

# ===================== 测试函数 =====================

def test_loss(model, dataset, device, batch_size=64):
    """① 算验证集 Loss 和 PPL"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size].to(device)
            if batch.size(0) == 0: continue
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, model.cfg.vocab_size),
                batch[:, 1:].reshape(-1),
                ignore_index=0  # 忽略 padding
            )
            total_loss += loss.item() * batch[:, 1:].ne(0).sum().item()
            total_tokens += batch[:, 1:].ne(0).sum().item()
    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 10 else float('inf')
    return avg_loss, ppl


def test_task_accuracy(model, dataset, device, task_name, task_token, num_tests=50):
    """
    ② 测试单个任务的准确率
    找到 → 后面的答案位置，逐个 token 比对
    """
    model.eval()
    correct_sequences = 0
    total_sequences = 0
    correct_tokens = 0
    total_tokens = 0

    with torch.no_grad():
        for _ in range(num_tests):
            # 找一个该任务的样本
            for attempt in range(100):
                idx = random.randint(0, len(dataset) - 1)
                sample = dataset[idx]
                if sample[0].item() == task_token:
                    break
            else:
                continue

            ids = sample.unsqueeze(0).to(device)
            logits = model(ids[:, :-1])  # [1, T-1, vocab]

            # 找 → 的位置
            arrow_pos = (ids[0] == 2).nonzero()
            if len(arrow_pos) == 0:
                continue

            ans_start = arrow_pos[0].item() + 1  # → 后面第一个答案 token
            if ans_start >= ids.size(1) - 1:
                continue

            # 逐 token 比较答案
            seq_correct = True
            for j in range(ans_start, ids.size(1) - 1):
                if ids[0, j].item() == 0:  # 遇到 padding 停止
                    break
                pred = logits[0, j - 1].argmax().item()  # j-1 因为 logits 是输入到下一个的
                true = ids[0, j].item()
                if pred == true:
                    correct_tokens += 1
                else:
                    seq_correct = False
                total_tokens += 1

            if seq_correct and total_tokens > 0:
                correct_sequences += 1
            total_sequences += 1

    model.train()
    seq_acc = correct_sequences / max(total_sequences, 1) * 100
    tok_acc = correct_tokens / max(total_tokens, 1) * 100
    return seq_acc, tok_acc


def interactive_generate(model, device, cfg, max_new=16):
    """
    ③ 交互式：输入一个序列，模型自回归生成
    """
    model.eval()
    print("\n" + "=" * 60)
    print(" 交互模式（输入 'quit' 退出）")
    print("格式示例：")
    print("  COPY:  1,17,42,2")
    print("  ADD:   3,15,4,25,5")
    print("  REV:   6,10,20,30,2")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n输入 token 序列（逗号分隔）: ").strip()
        except EOFError:
            break
        if user_input.lower() in ('quit', 'exit', 'q'):
            break
        if not user_input:
            continue

        # 解析输入
        try:
            tokens = [int(x.strip()) for x in user_input.split(',')]
        except ValueError:
            print(" 格式错误，请用逗号分隔的数字")
            continue

        # 补齐/截断到 seq_len
        if len(tokens) > cfg.seq_len:
            tokens = tokens[:cfg.seq_len]
        while len(tokens) < cfg.seq_len:
            tokens.append(0)

        input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

        # 自回归生成
        generated = tokens.copy()
        with torch.no_grad():
            for _ in range(max_new):
                # 找最后一个非 padding 位置
                valid_len = len([t for t in generated if t != 0])
                if valid_len >= cfg.seq_len:
                    break
                inp = torch.tensor([generated[:valid_len] + [0] * (cfg.seq_len - valid_len)], dtype=torch.long).to(device)
                logits = model(inp[:, :-1])
                # 预测下一个 token（看最后一个有效位置）
                next_token = logits[0, valid_len - 1].argmax().item()
                generated[valid_len] = next_token
                if next_token == 0:
                    break

        # 提取答案部分
        arrow_idx = None
        for i, t in enumerate(generated):
            if t == 2:  # → token
                arrow_idx = i
                break

        print(f"\n  输入: {tokens[:valid_len]}")
        if arrow_idx is not None:
            answer = generated[arrow_idx + 1:]
            answer = [t for t in answer if t != 0]
            print(f"  模型输出: {answer}")
        else:
            print(f"  完整生成: {[t for t in generated if t != 0]}")

    model.train()
    print(" 退出交互模式")


# ===================== Main =====================
def main():
    # ---- 参数 ----
    checkpoint_path = "./checkpoints/minigpt_step_500.pt"
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]

    if not os.path.exists(checkpoint_path):
        print(f"   找不到 checkpoint: {checkpoint_path}")
        print(f"   请先训练模型或指定正确路径: python test_minigpt.py ./checkpoints/minigpt_step_500.pt")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = Config()

    # ---- 加载模型 ----
    print(f" 加载模型: {checkpoint_path}")
    model = MiniGPT(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f" 模型加载成功 | Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f" Device: {device}")
    print()

    # ---- ① 验证集 Loss ----
    print("=" * 60)
    print("① 验证集 Loss / PPL")
    print("=" * 60)
    val_ds = SyntheticDataset(500, cfg.max_seq_len , cfg.vocab_size)
    val_loss, val_ppl = test_loss(model, val_ds, device)
    print(f"  val_loss = {val_loss:.4f}")
    print(f"  val_ppl  = {val_ppl:.2f}")
    print()

    # ---- ② 三种任务准确率 ----
    print("=" * 60)
    print("② 任务准确率测试（每种 50 个样本）")
    print("=" * 60)

    tasks = [
        ("COPY", 1),
        ("ADD", 3),
        ("REV", 6),
    ]

    for task_name, task_token in tasks:
        seq_acc, tok_acc = test_task_accuracy(model, val_ds, device, task_name, task_token, num_tests=50)
        print(f"  {task_name:6s}: 序列准确率={seq_acc:5.1f}% | Token准确率={tok_acc:5.1f}%")

    print()

    # ---- ③ 交互模式 ----
    interactive_generate(model, device, cfg)

    print("\n 测试完成！")


if __name__ == "__main__":
    main()