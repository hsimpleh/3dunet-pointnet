# train.py (GPU/CPU 自动适配 + 每 epoch 保存 + 自动续训练)
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ========= 保证本项目可 import =========
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# ========= 项目内模块 =========
from train_test_utils.dataloader import RadarLidarDataset
from train_test_utils.losses import MultiTaskLoss
from models.radarhd_unet3d import Radar3DUNet

# ===================== 配置 =====================
DATASET_DIR = "D:/毕业设计/RedarHD-3D/create_dataset/dataset_unet"
SEQ_LIST = [f"S{i}" for i in range(1, 4)]
BATCH_SIZE = 1
EPOCHS = 20
LEARNING_RATE = 1e-4
VAL_RATIO = 0.1
NUM_WORKERS = 2
CHECKPOINT_DIR = "checkpoints"
PROGRESS_FILE = os.path.join(CHECKPOINT_DIR, "progress.txt")  # 记录最新训练 epoch
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ----------------- 设备 -----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", DEVICE)


# ===================== 训练一个 epoch =====================
def train_one_epoch(epoch, model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", ncols=120)

    for step, batch in enumerate(pbar):
        radar = batch["radar"].to(DEVICE)
        occ_gt = batch["occ"].to(DEVICE)
        hgt_gt = batch["height"].to(DEVICE)

        optimizer.zero_grad()
        occ_pred, hgt_pred = model(radar)

        if occ_pred.shape[-2:] != occ_gt.shape[-2:]:
            occ_pred = torch.nn.functional.interpolate(
                occ_pred, size=occ_gt.shape[-2:], mode='bilinear', align_corners=False
            )
            hgt_pred = torch.nn.functional.interpolate(
                hgt_pred, size=hgt_gt.shape[-2:], mode='bilinear', align_corners=False
            )

        loss_dict = criterion(occ_pred, hgt_pred, occ_gt, hgt_gt)
        loss = loss_dict["total"]

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        pbar.set_postfix({
            "total": f"{loss.item():.4f}",
            "bce": f"{loss_dict['bce']:.3f}",
            "dice": f"{loss_dict['dice']:.3f}",
            "hgt": f"{loss_dict['height']:.3f}"
        })

    return running_loss / len(loader)


# ===================== 验证 =====================
@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0

    for batch in loader:
        radar = batch["radar"].to(DEVICE)
        occ_gt = batch["occ"].to(DEVICE)
        hgt_gt = batch["height"].to(DEVICE)

        occ_pred, hgt_pred = model(radar)

        if occ_pred.shape[-2:] != occ_gt.shape[-2:]:
            occ_pred = torch.nn.functional.interpolate(
                occ_pred, size=occ_gt.shape[-2:], mode='bilinear', align_corners=False
            )
            hgt_pred = torch.nn.functional.interpolate(
                hgt_pred, size=hgt_gt.shape[-2:], mode='bilinear', align_corners=False
            )

        loss_dict = criterion(occ_pred, hgt_pred, occ_gt, hgt_gt)
        running_loss += loss_dict["total"].item()

    return running_loss / len(loader)


# ===================== 主函数 =====================
def main():
    # -------- Dataset --------
    full_dataset = RadarLidarDataset(DATASET_DIR, SEQ_LIST)
    print(f"📦 Total samples: {len(full_dataset)}")

    val_size = int(len(full_dataset) * VAL_RATIO)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    # -------- Model / Loss / Optimizer --------
    model = Radar3DUNet(in_channels=1, base_ch=16, use_temporal_attention=False).to(DEVICE)
    criterion = MultiTaskLoss(height_weight=0.3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # -------- 读取上次训练进度 --------
    start_epoch = 1
    latest_ckpt = None
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            start_epoch = int(f.read().strip()) + 1  # 下一轮 epoch
        # 找到对应模型文件
        latest_ckpt = os.path.join(CHECKPOINT_DIR, f"model_epoch{start_epoch-1}_3T.pth")
        if os.path.exists(latest_ckpt):
            print(f"📥 Loading checkpoint: {latest_ckpt}")
            model.load_state_dict(torch.load(latest_ckpt, map_location=DEVICE))

    # -------- Training Loop --------
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\n====== Epoch {epoch}/{EPOCHS} ======")
        train_loss = train_one_epoch(epoch, model, train_loader, optimizer, criterion)
        print(f"🟢 Train loss: {train_loss:.4f}")

        val_loss = validate(model, val_loader, criterion)
        print(f"🟡 Val loss  : {val_loss:.4f}")

        # -------- 保存模型 --------
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch{epoch}_3T.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

        # -------- 更新进度文件 --------
        with open(PROGRESS_FILE, "w") as f:
            f.write(str(epoch))

    print("\n✅ Training finished!")


if __name__ == "__main__":
    main()
