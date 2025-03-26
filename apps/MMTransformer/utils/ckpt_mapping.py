import torch

state_dict = torch.load(
    "/dev/shm/consolidated.pth",
    weights_only=True,
)
filtered_state_dict = {
    key.replace("gen_transformer.", ""): value
    for key, value in state_dict["model"].items()
    if key.startswith("gen_transformer.")
}
torch.save(
    filtered_state_dict, "/jfs/checkpoints/gen_transformer-1B/Imagenet-400K/ckpt.pth"
)
