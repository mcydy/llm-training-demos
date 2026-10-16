"""
╔════════════════════════════════════════════════════════════════════════════╗
║  mini_gpt_v2/test.py — 加载训练好的模型，全面测试 + 交互生成                ║
║  运行：cd mini_gpt_v2 && python test.py [checkpoint_path]                  ║
║  默认加载 ./checkpoints/minigpt_final.pt                                   ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════════════
# 0. 配置（测试时 dropout=0）
# ══════════════════════════════════════════════════════════════════════════════
class Config:
    vocab_size    = 128
    hidden_size   = 256
    num_layers    = 6
    num_heads     = 8
    ff_multiplier = 4
    max_seq_len   = 64
    dropout       = 0.0  # 推理时关闭

cfg = Config()

# 任务常量（必须和训练时一致）
PAD, BOS, EOS, SEP = 0, 1, 2, 7
COPY, ADD, REV = 3, 4, 5
NUM_START = 10  # token 10 = 数字 0

# ══════════════════════════════════════════════════════════════════════════════
# 1. 模型定义（和训练时完全一致）
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
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
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
# 2. 数据集（和训练时一致，用于公平评估）
# ══════════════════════════════════════════════════════════════════════════════
class SyntheticDataset:
    """直接生成样本（不继承 Dataset，测试时不需要 DataLoader）"""
    PAD, BOS, EOS, SEP = 0, 1, 2, 7
    COPY, ADD, REV = 3, 4, 5
    NUM_START = 10

    def __init__(self, num_samples, seq_len):
        self.seq_len = seq_len
        self.samples = [self._make() for _ in range(num_samples)]

    def _make(self):
        L = self.seq_len
        kind = random.choice([0, 1, 2])
        if kind == 0:
            n = random.randint(3, 6)
            nums = [random.randint(0, 9) for _ in range(n)]
            body = [self.NUM_START + x for x in nums]
            answer = body[:]
            tag = self.COPY
        elif kind == 1:
            a = random.randint(0, 9)
            b = random.randint(0, 9)
            s = a + b
            body = [self.NUM_START + a, self.NUM_START + b]
            answer = [self.NUM_START + s]
            tag = self.ADD
        else:
            n = random.randint(3, 6)
            nums = [random.randint(0, 9) for _ in range(n)]
            body = [self.NUM_START + x for x in nums]
            answer = [self.NUM_START + x for x in reversed(nums)]
            tag = self.REV
        seq = [tag] + body + [self.SEP] + answer + [self.EOS]
        if len(seq) > L: seq = seq[:L]
        while len(seq) < L: seq.append(self.PAD)
        return torch.tensor(seq, dtype=torch.long)

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

# ══════════════════════════════════════════════════════════════════════════════
# 3. 推理工具
# ══════════════════════════════════════════════════════════════════════════════
def generate(model, device, prompt_tokens, max_new=16):
    """贪心自回归生成，遇到 EOS 或 PAD 停止"""
    model.eval()
    with torch.no_grad():
        x = torch.tensor([prompt_tokens], device=device)
        for _ in range(max_new):
            logits = model(x)[:, -1, :]
            nxt = logits.argmax(-1, keepdim=True)
            nid = nxt.item()
            if nid == EOS or nid == PAD:
                break
            x = torch.cat([x, nxt], dim=1)
    model.train()
    # 返回新生成的部分（去掉 prompt）
    new_tokens = x[0, len(prompt_tokens):].tolist()
    return [t for t in new_tokens if t not in (PAD, EOS)]

def decode_nums(tokens):
    """token id → 数字字符串列表"""
    return [str(t - NUM_START) for t in tokens if t >= NUM_START]

def encode_nums(nums):
    """数字列表 → token id 列表"""
    return [n + NUM_START for n in nums]

# ══════════════════════════════════════════════════════════════════════════════
# 4. 评估函数
# ══════════════════════════════════════════════════════════════════════════════
def find_sample(dataset, task_tag, max_attempts=500):
    """从数据集中找一个指定任务的样本"""
    for _ in range(max_attempts):
        idx = random.randint(0, len(dataset) - 1)
        if dataset[idx][0].item() == task_tag:
            return dataset[idx]
    return None

def get_prompt_and_answer(sample):
    """从样本中提取 prompt（到 SEP）和真实答案"""
    sep_positions = (sample == SEP).nonzero()
    if len(sep_positions) == 0:
        return None, None
    sep_idx = sep_positions[0].item()
    prompt = sample[:sep_idx + 1].tolist()  # 包含 SEP
    true_answer = []
    for i in range(sep_idx + 1, len(sample)):
        val = sample[i].item()
        if val in (PAD, EOS): break
        true_answer.append(val)
    return prompt, true_answer

def test_task_accuracy(model, dataset, device, task_tag, num_tests=100):
    """测试单个任务的准确率（逐 token 完全匹配才算对）"""
    model.eval()
    correct = 0
    total = 0
    for _ in range(num_tests):
        sample = find_sample(dataset, task_tag)
        if sample is None: continue
        prompt, true_answer = get_prompt_and_answer(sample)
        if prompt is None: continue

        pred = generate(model, device, prompt, max_new=len(true_answer) + 3)

        total += 1
        if pred == true_answer:
            correct += 1

    model.train()
    return correct, total, (correct / max(total, 1)) * 100

def test_token_level_accuracy(model, dataset, device, task_tag, num_tests=100):
    """逐 token 级别的准确率（更细粒度）"""
    model.eval()
    correct_tokens = 0
    total_tokens = 0
    for _ in range(num_tests):
        sample = find_sample(dataset, task_tag)
        if sample is None: continue
        prompt, true_answer = get_prompt_and_answer(sample)
        if prompt is None or len(true_answer) == 0: continue

        pred = generate(model, device, prompt, max_new=len(true_answer) + 3)

        # 逐 token 比较（按最短长度）
        min_len = min(len(pred), len(true_answer))
        for i in range(min_len):
            total_tokens += 1
            if pred[i] == true_answer[i]:
                correct_tokens += 1
        # 长度不一致的部分算错
        total_tokens += abs(len(pred) - len(true_answer))

    model.train()
    return correct_tokens, total_tokens, (correct_tokens / max(total_tokens, 1)) * 100

def show_samples(model, dataset, device, task_tag, task_name, num_show=5):
    """展示几个具体样本的预测结果"""
    print(f"\n  ┌─ {task_name} 样本详情 ─────────────────────────────────────")
    for i in range(num_show):
        sample = find_sample(dataset, task_tag)
        if sample is None: continue
        prompt, true_answer = get_prompt_and_answer(sample)
        if prompt is None: continue

        pred = generate(model, device, prompt, max_new=len(true_answer) + 3)

        prompt_nums = decode_nums([t for t in prompt if t >= NUM_START])
        true_nums = decode_nums(true_answer)
        pred_nums = decode_nums(pred)

        match = "✅" if pred == true_answer else "❌"
        print(f"  │  [{i+1}] {match} prompt={prompt_nums}")
        print(f"  │       true={true_nums} | pred={pred_nums}")
    print(f"  └──────────────────────────────────────────────────────────────")

# ══════════════════════════════════════════════════════════════════════════════
# 5. 交互模式
# ══════════════════════════════════════════════════════════════════════════════
def interactive_mode(model, device):
    """交互式测试：手动输入或随机生成"""
    print()
    print("=" * 64)
    print("🎮 交互模式（输入 'quit' 退出，'help' 查看帮助）")
    print("=" * 64)

    while True:
        try:
            user = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in ('quit', 'exit', 'q'):
            break
        if not user:
            continue
        if user.lower() == 'help':
            _print_help()
            continue

        # 快捷命令
        if user.lower() == 'copy':
            nums = [random.randint(0, 9) for _ in range(random.randint(3, 6))]
            prompt = [COPY] + encode_nums(nums) + [SEP]
            task_name = f"COPY {nums}"
        elif user.lower() == 'add':
            a, b = random.randint(0, 9), random.randint(0, 9)
            prompt = [ADD, a + NUM_START, b + NUM_START, SEP]
            task_name = f"ADD {a}+{b}={a+b}"
        elif user.lower() == 'rev':
            nums = [random.randint(0, 9) for _ in range(random.randint(3, 6))]
            prompt = [REV] + encode_nums(nums) + [SEP]
            task_name = f"REV {nums} → {list(reversed(nums))}"
        else:
            # 手动输入数字序列（逗号分隔）
            try:
                nums = [int(x.strip()) for x in user.split(',')]
                # 判断类型
                if len(nums) >= 3 and nums[0] in (COPY, ADD, REV):
                    prompt = nums  # 直接用作 token
                    task_name = f"manual {nums}"
                else:
                    # 默认当 COPY
                    prompt = [COPY] + encode_nums(nums) + [SEP]
                    task_name = f"COPY {nums}"
            except ValueError:
                print("  ❌ 格式错误，输入 'help' 查看用法")
                continue

        # 生成
        pred = generate(model, device, prompt, max_new=12)
        pred_nums = decode_nums(pred)
        prompt_nums = decode_nums([t for t in prompt if t >= NUM_START])

        print(f"  task:   {task_name}")
        print(f"  prompt: {prompt_nums}")
        print(f"  output: {pred_nums}")

    print("\n👋 再见！")

def _print_help():
    print()
    print("  用法：")
    print("    copy          → 随机生成 COPY 任务")
    print("    add           → 随机生成 ADD 任务")
    print("    rev           → 随机生成 REV 任务")
    print("    3,7,5,2       → COPY 任务：复制 3 7 5 2")
    print("    4,2,6          → ADD 任务：2+6=8（token 格式）")
    print("    quit           → 退出")
    print()
    print("  Token 速查：")
    print("    3=COPY  4=ADD  5=REV  7=SEP")
    print("    数字 token = 实际数字 + 10（如 15 → 数字 5）")

# ══════════════════════════════════════════════════════════════════════════════
# 6. 主函数
# ══════════════════════════════════════════════════════════════════════════════
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 确定 checkpoint 路径 ──────────────────────────────────────────────
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    else:
        # 自动找最新的
        ckpt_dir = "./checkpoints"
        if os.path.isdir(ckpt_dir):
            files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
            if files:
                # 优先 final，否则取最新的 step
                final = [f for f in files if 'final' in f]
                if final:
                    ckpt_path = os.path.join(ckpt_dir, final[0])
                else:
                    step_files = [f for f in files if f.startswith('minigpt_step_')]
                    if step_files:
                        step_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                        ckpt_path = os.path.join(ckpt_dir, step_files[-1])
                    else:
                        ckpt_path = os.path.join(ckpt_dir, files[-1])
            else:
                ckpt_path = "./checkpoints/minigpt_final.pt"
        else:
            ckpt_path = "./checkpoints/minigpt_final.pt"

    if not os.path.exists(ckpt_path):
        print(f"❌ 找不到 checkpoint: {ckpt_path}")
        print(f"   请先运行: python train.py")
        print(f"   或指定路径: python test.py ./checkpoints/minigpt_step_0500.pt")
        return

    # ── 加载模型 ──────────────────────────────────────────────────────────
    print(f"📂 加载: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    # 用保存的 config 重建
    saved_cfg = ckpt.get('config', {})
    for k, v in saved_cfg.items():
        setattr(cfg, k, v)
    cfg.dropout = 0.0  # 推理时关闭

    model = MiniGPT(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型加载成功")
    print(f"   Params: {n_params/1e6:.2f}M | Device: {device}")
    print(f"   Arch: {cfg.num_layers}L × {cfg.hidden_size}d, {cfg.num_heads} heads")
    if 'val_loss' in ckpt:
        print(f"   val_loss={ckpt['val_loss']:.4f} | val_ppl={ckpt['val_ppl']:.2f}")
    print()

    # ── 创建验证集（和训练时同分布）──────────────────────────────────────
    random.seed(123)  # 固定种子，保证可复现
    val_set = SyntheticDataset(num_samples=2000, seq_len=cfg.max_seq_len)

    # ═══════════════════════════════════════════════════════════════════════
    # ① 综合评估
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 64)
    print("📊 ① 综合准确率评估（每种任务 100 个样本）")
    print("=" * 64)

    tasks = [("COPY", COPY), ("ADD", ADD), ("REV", REV)]
    results = {}

    for name, tag in tasks:
        # 序列准确率
        corr, total, seq_acc = test_task_accuracy(model, val_set, device, tag, num_tests=100)
        # Token 准确率
        corr_t, total_t, tok_acc = test_token_level_accuracy(model, val_set, device, tag, num_tests=100)

        results[name] = (seq_acc, tok_acc)
        bar_seq = "█" * int(seq_acc / 5) + "░" * (20 - int(seq_acc / 5))
        bar_tok = "█" * int(tok_acc / 5) + "░" * (20 - int(tok_acc / 5))

        print(f"  {name:6s} 序列准确率: {seq_acc:5.1f}% {bar_seq}  ({corr}/{total})")
        print(f"  {'':6s} Token准确率: {tok_acc:5.1f}% {bar_tok}  ({corr_t}/{total_t})")

    # 总评
    avg_seq = sum(r[0] for r in results.values()) / 3
    avg_tok = sum(r[1] for r in results.values()) / 3
    print(f"\n  ─────────────────────────────────────")
    print(f"  平均序列准确率: {avg_seq:5.1f}%")
    print(f"  平均Token准确率: {avg_tok:5.1f}%")

    # 评级
    if avg_seq >= 95:
        grade = "🏆 优秀！模型几乎完美掌握所有任务"
    elif avg_seq >= 80:
        grade = "👍 良好，大部分任务已学会"
    elif avg_seq >= 50:
        grade = "⚠️ 一般，部分任务有掌握但不稳定"
    else:
        grade = "❌ 未学会，建议检查训练流程或加长训练"
    print(f"  {grade}")

    # ═══════════════════════════════════════════════════════════════════════
    # ② 样本详情
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*64}")
    print("🔍 ② 样本详情（看模型具体输出）")
    print(f"{'='*64}")

    for name, tag in tasks:
        show_samples(model, val_set, device, tag, name, num_show=5)

    # ═══════════════════════════════════════════════════════════════════════
    # ③ 难度递增测试（COPY 长度对准确率的影响）
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*64}")
    print("📈 ③ COPY 任务：长度 vs 准确率")
    print(f"{'='*64}")

    model.eval()
    for length in [3, 4, 5, 6]:
        correct = 0
        total = 0
        for _ in range(50):
            nums = [random.randint(0, 9) for _ in range(length)]
            prompt = [COPY] + encode_nums(nums) + [SEP]
            true_answer = encode_nums(nums)
            pred = generate(model, device, prompt, max_new=length + 2)
            total += 1
            if pred == true_answer:
                correct += 1
        acc = correct / max(total, 1) * 100
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"  COPY len={length}: {acc:5.1f}% {bar}  ({correct}/{total})")

    # ═══════════════════════════════════════════════════════════════════════
    # ④ ADD 难度测试（数字大小对准确率的影响）
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*64}")
    print("📈 ④ ADD 任务：不同数字范围 vs 准确率")
    print(f"{'='*64}")

    for max_val in [5, 9, 15, 18]:
        correct = 0
        total = 0
        for _ in range(50):
            a = random.randint(0, max_val)
            b = random.randint(0, max_val)
            s = a + b
            if s > 18: continue  # 超出词表
            prompt = [ADD, a + NUM_START, b + NUM_START, SEP]
            true_answer = [s + NUM_START]
            pred = generate(model, device, prompt, max_new=3)
            total += 1
            if pred == true_answer:
                correct += 1
        acc = correct / max(total, 1) * 100
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"  ADD 0~{max_val:2d}: {acc:5.1f}% {bar}  ({correct}/{total})")

    # ═══════════════════════════════════════════════════════════════════════
    # ⑤ 进入交互模式
    # ═══════════════════════════════════════════════════════════════════════
    print()
    ans = input("🎮 进入交互模式？(y/n): ").strip().lower()
    if ans in ('y', 'yes'):
        interactive_mode(model, device)

    print("\n✅ 测试完成！")

if __name__ == "__main__":
    main()
