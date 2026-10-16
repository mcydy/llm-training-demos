# 项目名称

本项目是个人学习和整理 PyTorch 训练、DeepSpeed 训练以及验证相关脚本。当前目录中每个文件/文件夹的作用如下。

## 目录说明

| 名称 | 作用 |
|---|---|
| `train_pytorch_dryrun.py`  | PyTorch 极简训练脚本，空跑，演示模型训练流程 |
| `train_pytorch_minigpt_basic.py` | 4层 Transformer Block、隐藏层维度128 |
| `train_pytorch_minigpt_plus.py` |  4层 Transformer Block、隐藏层维度128，增加前向传播过程中每次forward日志等 |
| `train_pytorch_minigpt_pro.py` | 6层 Transformer Block，隐藏层维度256 |
| `train_pytorch_minigpt_promax.py` | 6层 Transformer Block，隐藏层维度256，增加课程学习（Curriculum Learning），提升准确率 |
| `train_deepspeed_dryrun.py` | DeepSpeed 训练优化， 空跑，演示Deepspeed训练流程 |
| `train_deepspeed_minigpt.py` | DeepSpeed 训练优化， 增加课程学习等 |
| `train_verify_plus.py` | 模型验证脚本 |
| `train_verify_promax.py` | 模型验证脚本 |
| `train_pt2safetensors.py` | pt转换safetensors脚本
