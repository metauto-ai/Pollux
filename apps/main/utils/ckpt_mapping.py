import torch

state_dict = torch.load(
    "/jfs/checkpoints/dump/Pollux_v0.7_gen_transformer_preliminary/checkpoints/0000327500/consolidated/consolidated.pth",
    weights_only=True,
)
print(123)
for k, v in state_dict["model"].items():
    print(k, v.shape)
