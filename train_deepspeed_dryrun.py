
# 使用deepspeed官方启动器,避免mpi4py问题
# deepspeed --num_gpus=1 train_deepspeed_demo.py

import os
import torch
import deepspeed
from torch.utils.data import Dataset, DataLoader
from torch import nn

# ============================================================
# 1. 极简因果 LM 模型
# ============================================================
class TinyLM(nn.Module):
    def __init__(self, vocab=256, d_model=128, n_heads=4, n_layers=2, max_len=64):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=256,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        x = self.tok_emb(input_ids) + self.pos_emb[:, :T]
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.encoder(x, mask=mask)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return {"logits": logits, "loss": loss}


# ============================================================
# 2. 假数据
# ============================================================
class RandomTextDataset(Dataset):
    def __init__(self, n_samples=2000, seq_len=64, vocab=256):
        self.n = n_samples
        self.seq_len = seq_len
        self.vocab = vocab

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        ids = torch.randint(0, self.vocab, (self.seq_len,))
        return ids[:-1], ids[1:]


# ============================================================
# 3. DeepSpeed 配置
# ============================================================
ds_config = {
    "train_micro_batch_size_per_gpu": 16,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 3e-3, "weight_decay": 0.01}
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"},
        "allgather_partitions": True,
        "reduce_scatter": True,
        "overlap_comm": True
    },
    "steps_per_print": 10
}


# ============================================================
# 4. 主函数（修正：def main():）
# ============================================================
def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    model = TinyLM(vocab=256, d_model=128, n_heads=4, n_layers=2, max_len=64)

    dataset = RandomTextDataset(n_samples=2000, seq_len=64, vocab=256)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    print(f"[rank {local_rank}] 开始训练，参数总量: {sum(p.numel() for p in model.parameters())/1e3:.1f}K")

    for epoch in range(3):
        for step, (inp, lbl) in enumerate(loader):
            inp = inp.cuda(local_rank)
            lbl = lbl.cuda(local_rank)

            outputs = model_engine(inp, labels=lbl)
            loss = outputs["loss"]

            model_engine.backward(loss)
            model_engine.step()

            if step % 10 == 0 and local_rank == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

    if local_rank == 0:
        model_engine.save_checkpoint("./deepspeed_ckpt")
        print(" 检查点已保存到 ./deepspeed_ckpt")


if __name__ == "__main__":
    # deepspeed.init_distributed(dist_backend="nccl")
    deepspeed.init_distributed()
    main()



