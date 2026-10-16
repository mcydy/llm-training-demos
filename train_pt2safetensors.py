from safetensors.torch import save_file
import torch

ckpt = torch.load("./checkpoints/minigpt_step_500.pt", map_location="cpu", weights_only=True)
state_dict = ckpt['model_state_dict']

# 断开共享
state_dict['lm_head.weight'] = state_dict['lm_head.weight'].clone()

# 保存
save_file(state_dict, "./checkpoints/minigpt_step_500.safetensors")
print(" 转换完成")