import os
import glob
import pickle
import torch
from torch.utils.data import Dataset


class RadarLidarDataset(Dataset):
    """
    读取 3D UNet 雷达增强数据

    radar        : (T, H, W)
    lidar_occ    : (H, W)
    lidar_height : (H, W)
    """

    def __init__(
        self,
        dataset_dir,
        seq_list,
        split="train",
        split_ratio=(0.7, 0.15, 0.15),
        seed=42,
        num_frames=10
    ):

        assert split in ["train", "val", "test"]
        assert abs(sum(split_ratio) - 1.0) < 1e-6

        self.num_frames = num_frames
        self.split = split

        all_samples = []

        for seq in seq_list:
            seq_dir = os.path.join(dataset_dir, seq)
            pkl_files = sorted(glob.glob(os.path.join(seq_dir, "*.pkl")))
            all_samples.extend(pkl_files)

        assert len(all_samples) > 0, "❌ Dataset is empty"

        # 固定随机种子
        torch.manual_seed(seed)
        perm = torch.randperm(len(all_samples)).tolist()
        all_samples = [all_samples[i] for i in perm]

        # 数据集切分
        n_total = len(all_samples)
        n_train = int(split_ratio[0] * n_total)
        n_val = int(split_ratio[1] * n_total)

        if split == "train":
            self.samples = all_samples[:n_train]
        elif split == "val":
            self.samples = all_samples[n_train:n_train + n_val]
        else:
            self.samples = all_samples[n_train + n_val:]

        print(
            f"📦 {split.upper()} samples: {len(self.samples)} | "
            f"T = {self.num_frames}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        with open(self.samples[idx], "rb") as f:
            sample = pickle.load(f)

        radar = torch.tensor(sample["radar"], dtype=torch.float32)        # (T,H,W)
        occ = torch.tensor(sample["lidar_occ"], dtype=torch.float32)      # (H,W)
        hgt = torch.tensor(sample["lidar_height"], dtype=torch.float32)   # (H,W)

        # ---------------- Radar normalization ----------------
        # 毫米波雷达通常非常稀疏，用 log scaling 更稳定
        radar = torch.log1p(radar * 10.0)

        # ---------------- Temporal sampling ----------------
        if self.num_frames is not None:

            T = radar.shape[0]

            # 训练时随机采样时间窗口
            if self.split == "train" and T > self.num_frames:

                start = torch.randint(
                    0,
                    T - self.num_frames + 1,
                    (1,)
                ).item()

                radar = radar[start:start + self.num_frames]

            # 验证 / 测试取最后几帧
            else:

                if T >= self.num_frames:
                    radar = radar[-self.num_frames:]
                else:
                    pad = self.num_frames - T
                    radar = torch.cat(
                        [radar, radar[-1:].repeat(pad,1,1)],
                        dim=0
                    )

        # ---------------- Network input format ----------------
        radar = radar.unsqueeze(0)   # (1,T,H,W)
        occ = occ.unsqueeze(0)       # (1,H,W)
        hgt = hgt.unsqueeze(0)       # (1,H,W)

        return {
            "radar": radar,
            "occ": occ,
            "height": hgt
        }