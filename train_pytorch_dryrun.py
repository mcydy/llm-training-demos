import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

# 构造假数据：10000 个样本，输入 512 维，二分类
X = torch.randn(10000, 512)
y = (X.sum(dim=1) > 0).long()

loader = DataLoader(TensorDataset(X, y), batch_size=256, shuffle=True)

# 简单 MLP
model = nn.Sequential(
    nn.Linear(512, 1024), nn.ReLU(),
    nn.Linear(1024, 512),  nn.ReLU(),
    nn.Linear(512, 2),
).cuda()

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

model.train()
t0 = time.time()
for epoch in range(5):
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.cuda(), yb.cuda()
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    print(f"epoch {epoch+1}: loss={total/n:.4f}  "
          f"显存={torch.cuda.memory_allocated()/1024**3:.2f}GB")

print(f"训练耗时: {time.time()-t0:.1f}s")
print("GPU:", torch.cuda.get_device_name(0))
