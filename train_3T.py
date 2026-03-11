import os
import sys
import gc
import re
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ===================== 路径设置 =====================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from train_test_utils.dataloader import RadarLidarDataset
from train_test_utils.losses import MultiTaskLoss
from models.radarhd_unet3d import Radar3DUNet

DATASET_DIR = "/home/csw/Radar3DUNet/RedarHD-3D/create_dataset/dataset_unet"
SEQ_LIST = [f"S{i}" for i in range(1, 4)]

TOTAL_BATCH_SIZE = 4
EPOCHS = 60
LEARNING_RATE = 1e-4
VAL_RATIO = 0.1
NUM_WORKERS = 4

CHECKPOINT_DIR = "/home/csw/Radar3DUNet/RedarHD-3D/checkpoints_v4-10-1"
LOG_FILE = os.path.join(CHECKPOINT_DIR, "train_log.txt")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_COUNT = torch.cuda.device_count()
BATCH_SIZE = max(1, TOTAL_BATCH_SIZE // max(1, GPU_COUNT))

print(f"🚀 Using device: {DEVICE}, {GPU_COUNT} GPU(s) detected")
print(f"📦 Total batch size: {TOTAL_BATCH_SIZE}, per GPU: {BATCH_SIZE}")

torch.backends.cudnn.benchmark = True


# ===================== IoU计算 =====================
def compute_iou(pred, target):

    pred = (torch.sigmoid(pred) > 0.35).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return (intersection / (union + 1e-6)).item()


# ===================== MSE计算 =====================
def compute_mse(pred, target):

    mse = torch.mean((pred - target) ** 2)

    return mse.item()


# ===================== 检查点加载 =====================
def load_checkpoint_if_exists(model, optimizer, scaler):

    start_epoch = 1
    best_val = float("inf")

    ckpt_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("model_epoch")]

    if len(ckpt_files) == 0:
        print("⚪ No checkpoint found")
        return start_epoch, best_val

    def extract_epoch(name):
        match = re.search(r"epoch(\d+)", name)
        return int(match.group(1)) if match else -1

    ckpt_files.sort(key=extract_epoch)
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_files[-1])

    print(f"🔁 Resume training from {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = checkpoint["epoch"] + 1
    best_val = checkpoint.get("best_val", float("inf"))

    return start_epoch, best_val


# ===================== 训练 =====================
def train_one_epoch(epoch, model, loader, optimizer, criterion, scaler, log_f):

    model.train()
    running_loss = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", ncols=120)

    for step, batch in enumerate(pbar):

        radar = batch["radar"].to(DEVICE, non_blocking=True)
        occ_gt = batch["occ"].to(DEVICE, non_blocking=True)
        hgt_gt = batch["height"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):

            occ_pred, hgt_pred = model(radar)

            if occ_pred.shape[-2:] != occ_gt.shape[-2:]:

                occ_pred = torch.nn.functional.interpolate(
                    occ_pred, size=occ_gt.shape[-2:], mode="bilinear", align_corners=False
                )

                hgt_pred = torch.nn.functional.interpolate(
                    hgt_pred, size=hgt_gt.shape[-2:], mode="bilinear", align_corners=False
                )

            loss, focal, dice, h_loss = criterion(
                occ_pred, occ_gt, hgt_pred, hgt_gt
            )

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "focal": f"{focal:.3f}",
            "dice": f"{dice:.3f}",
            "hgt": f"{h_loss:.3f}"
        })

        log_f.write(
            f"Epoch {epoch}, Step {step}, "
            f"loss={loss.item():.4f}, "
            f"focal={focal:.3f}, "
            f"dice={dice:.3f}, "
            f"hgt={h_loss:.3f}\n"
        )

    return running_loss / len(loader)


# ===================== 验证 =====================
@torch.no_grad()
def validate(model, loader, criterion):

    model.eval()

    running_loss = 0
    iou_total = 0
    mse_total = 0
    count = 0

    for batch in loader:

        radar = batch["radar"].to(DEVICE)
        occ_gt = batch["occ"].to(DEVICE)
        hgt_gt = batch["height"].to(DEVICE)

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):

            occ_pred, hgt_pred = model(radar)

            if occ_pred.shape[-2:] != occ_gt.shape[-2:]:

                occ_pred = torch.nn.functional.interpolate(
                    occ_pred, size=occ_gt.shape[-2:], mode="bilinear", align_corners=False
                )

                hgt_pred = torch.nn.functional.interpolate(
                    hgt_pred, size=hgt_gt.shape[-2:], mode="bilinear", align_corners=False
                )

            loss, focal, dice, h_loss = criterion(
                occ_pred, occ_gt, hgt_pred, hgt_gt
            )

        running_loss += loss.item()

        iou_total += compute_iou(occ_pred, occ_gt)
        mse_total += compute_mse(hgt_pred, hgt_gt)

        count += 1

    return (
        running_loss / len(loader),
        iou_total / count,
        mse_total / count
    )


# ===================== 主函数 =====================
def main():

    torch.cuda.empty_cache()

    log_f = open(LOG_FILE, "a", encoding="utf-8")

    dataset = RadarLidarDataset(DATASET_DIR, SEQ_LIST)
    print(f"📦 Total samples: {len(dataset)}")

    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    print("\n========== 输入数据检查 ==========")

    sample_batch = next(iter(train_loader))

    radar_sample = sample_batch["radar"]
    occ_sample = sample_batch["occ"]
    hgt_sample = sample_batch["height"]

    print("Radar shape:", radar_sample.shape)
    print("Occupancy shape:", occ_sample.shape)
    print("Height shape:", hgt_sample.shape)

    B, C, T, H, W = radar_sample.shape

    print("\n===== 网络输入维度 =====")
    print(f"Batch size (B): {B}")
    print(f"Channels (C): {C}")
    print(f"Temporal frames (T): {T}")
    print(f"Height (H): {H}")
    print(f"Width (W): {W}")

    print(f"\n最终输入格式: (B,C,T,H,W) = ({B},{C},{T},{H},{W})")
    print("=================================\n")

    model = Radar3DUNet(in_channels=1, base_ch=128).to(DEVICE)

    if GPU_COUNT > 1:
        print(f"🔥 Using {GPU_COUNT} GPUs")
        model = torch.nn.DataParallel(model)

    criterion = MultiTaskLoss(height_weight=0.3).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    scaler = torch.amp.GradScaler()

    start_epoch, best_val = load_checkpoint_if_exists(model, optimizer, scaler)

    for epoch in range(start_epoch, EPOCHS + 1):

        print(f"\n====== Epoch {epoch}/{EPOCHS} ======")

        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, criterion, scaler, log_f
        )

        val_loss, val_iou, val_mse = validate(model, val_loader, criterion)

        scheduler.step()

        print(f"🟢 Train loss: {train_loss:.4f}")
        print(f"🟡 Val loss: {val_loss:.4f}")
        print(f"🔵 Val IoU : {val_iou:.4f}")
        print(f"🟣 Val MSE : {val_mse:.6f}")

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val": best_val
        }, os.path.join(CHECKPOINT_DIR, f"model_epoch{epoch}.pth"))

    print("\n✅ Training finished!")

    gc.collect()
    torch.cuda.empty_cache()

    log_f.close()


if __name__ == "__main__":

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    main()