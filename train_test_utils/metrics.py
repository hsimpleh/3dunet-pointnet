import torch
import torch.nn.functional as F

# =========================================================
# IoU (Intersection over Union) with mask
# =========================================================
@torch.no_grad()
def compute_iou(pred, target, occ_gt=None, thresh=0.5):
    """
    pred   : (B,1,H,W) logits
    target : (B,1,H,W) {0,1}
    occ_gt : (B,1,H,W) GT occupancy {0,1}, optional for masking

    return : mean IoU over valid samples
    """
    pred = torch.sigmoid(pred)
    pred = (pred > thresh).float()

    if occ_gt is not None:
        # ⚡ mask扩展，和训练loss保持一致
        mask = F.max_pool2d(occ_gt.float(), kernel_size=3, stride=1, padding=1) > 0.5
        pred = pred * mask
        target = target * mask

    # flatten to (B, H*W)
    pred = pred.flatten(1)
    target = target.flatten(1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection

    valid = union > 0
    if valid.sum() == 0:
        return 0.0

    iou = intersection[valid] / (union[valid] + 1e-6)
    return iou.mean().item()


# =========================================================
# Dice Score with mask
# =========================================================
@torch.no_grad()
def compute_dice(pred, target, occ_gt=None, thresh=0.5):
    """
    pred   : (B,1,H,W) logits
    target : (B,1,H,W) {0,1}
    occ_gt : (B,1,H,W) GT occupancy {0,1}, optional for masking

    return : mean Dice over valid samples
    """
    pred = torch.sigmoid(pred)
    pred = (pred > thresh).float()

    if occ_gt is not None:
        mask = F.max_pool2d(occ_gt.float(), kernel_size=3, stride=1, padding=1) > 0.5
        pred = pred * mask
        target = target * mask

    pred = pred.flatten(1)
    target = target.flatten(1)

    intersection = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)

    valid = denom > 0
    if valid.sum() == 0:
        return 0.0

    dice = 2.0 * intersection[valid] / (denom[valid] + 1e-6)
    return dice.mean().item()


# =========================================================
# Height MAE (Masked) with mask expansion
# =========================================================
@torch.no_grad()
def compute_height_mae(pred, target, occ_gt):
    """
    pred   : (B,1,H,W) predicted height
    target : (B,1,H,W) GT height
    occ_gt : (B,1,H,W) GT occupancy {0,1}

    return : mean MAE over occupied pixels
    """
    # ⚡ mask扩展，与loss保持一致
    mask = F.max_pool2d(occ_gt.float(), kernel_size=3, stride=1, padding=1) > 0.5

    if mask.sum() == 0:
        return 0.0

    mae = torch.abs(pred - target)
    return mae[mask].mean().item()


# =========================================================
# Quick test
# =========================================================
if __name__ == "__main__":
    B, H, W = 2, 128, 128
    occ_pred = torch.randn(B,1,H,W)          # logits
    hgt_pred = torch.rand(B,1,H,W)           # sigmoid output
    occ_gt = torch.randint(0,2,(B,1,H,W)).float()
    hgt_gt = torch.rand(B,1,H,W)

    print("IoU:", compute_iou(occ_pred, occ_gt, occ_gt))
    print("Dice:", compute_dice(occ_pred, occ_gt, occ_gt))
    print("Height MAE:", compute_height_mae(hgt_pred, hgt_gt, occ_gt))