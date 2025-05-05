import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward


class Loss:
    def __init__(self):
        self.dwt = DWTForward(J=1, mode='zero', wave='haar').cuda().to(torch.bfloat16)

    @staticmethod
    def mse_loss(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if isinstance(target, list) or isinstance(x, list):
            loss_list = [F.mse_loss(o, t.to(o.dtype)) for o, t in zip(x, target)]
            return torch.mean(torch.stack(loss_list))
        else:
            target = target.to(x.dtype)
            return F.mse_loss(x, target)

    @staticmethod
    def consine_loss_with_features(x: torch.Tensor, cond_l: torch.Tensor, 
                                    img_size: torch.Tensor, patch_size: int, target: torch.Tensor) -> torch.Tensor:
        pH = pW = patch_size
        H, W = img_size
        use_dynamic_res = isinstance(H, list) and isinstance(W, list)
        
        def _cosine_loss(x, t):
            return 1 - F.cosine_similarity(x, t.to(x.dtype), dim=-1).mean()

        if use_dynamic_res:
            # Extract features for each image based on its resolution
            return torch.stack([
                _cosine_loss(
                    x[i, cond_l[i]:cond_l[i] + (_H // pH) * (_W // pW)], 
                    target[i]
                )
                for i, (_H, _W) in enumerate(zip(H, W))
            ]).mean()
        else:
            # Handle condition length consistently whether it's a list or scalar
            offset = max(cond_l) if isinstance(cond_l, list) else cond_l
            feature_length = (H // pH) * (W // pW)
            img_features = x[:, offset:offset + feature_length]
            return _cosine_loss(img_features, target)
    
    
    def dwt_loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x_ll, x_h = self.dwt(x)
        x_lh, x_hl, x_hh = torch.unbind(x_h[0], dim=2)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        target_ll, target_h = self.dwt(target)
        target_lh, target_hl, target_hh = torch.unbind(target_h[0], dim=2)
        target = torch.cat([target_ll, target_lh, target_hl, target_hh], dim=1)
        return F.mse_loss(x, target)
