import os
import glob
import pickle
import numpy as np
import pandas as pd

# =========================
# 路径配置
# =========================
RADAR_DIR = "D:/毕业设计/RedarHD-3D/create_dataset/radar_dataset"
LIDAR_DIR = "D:/毕业设计/RedarHD-3D/data/lidar_pcl"
ALIGN_DIR = "D:/毕业设计/RedarHD-3D/create_dataset/time_alignment"
SAVE_DIR  = "D:/毕业设计/RedarHD-3D/create_dataset/dataset_unet"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# BEV 参数
# =========================
T = 20
X_MAX = 20.0
Y_MAX = 20.0
RES   = 0.1

H = int(2 * Y_MAX / RES)
W = int(X_MAX / RES)

# =========================
# Radar → BEV
# =========================
def radar_points_to_bev(points):
    """
    将单帧雷达点云投影为 BEV
    """
    bev = np.zeros((H, W), dtype=np.float32)

    if points is None:
        return bev

    points = np.asarray(points)
    if points.ndim != 2 or points.shape[0] == 0:
        return bev

    x, y = points[:, 0], points[:, 1]

    if points.shape[1] > 3:
        v = points[:, 3]
    else:
        v = np.ones_like(x)

    mask = (x > 0) & (x < X_MAX) & (np.abs(y) < Y_MAX)
    x, y, v = x[mask], y[mask], v[mask]

    ix = (x / RES).astype(np.int32)
    iy = ((y + Y_MAX) / RES).astype(np.int32)

    np.maximum.at(bev, (iy, ix), v)

    if bev.max() > 0:
        bev /= bev.max()

    return bev


# =========================
# LiDAR → Occupancy + Height（向量化）
# =========================
def lidar_to_bev_occ_height(points):
    """
    将 LiDAR 点云投影为：
    - 占据图 occ_map
    - 高度图 height_map
    """
    occ = np.zeros((H, W), dtype=np.float32)
    height = np.zeros((H, W), dtype=np.float32)

    if points.ndim != 2 or points.shape[0] == 0:
        return occ, height

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    mask = (x > 0) & (x < X_MAX) & (np.abs(y) < Y_MAX)
    x, y, z = x[mask], y[mask], z[mask]

    ix = (x / RES).astype(np.int32)
    iy = ((y + Y_MAX) / RES).astype(np.int32)

    occ[iy, ix] = 1.0
    np.maximum.at(height, (iy, ix), z)

    # 高度归一化（-2m ~ 2m）
    height = np.clip(height, -2.0, 2.0)
    height = (height + 2.0) / 4.0

    return occ, height


# =========================
# 主流程
# =========================
def process_sequence(seq):

    print("\n==============================")
    print("处理序列:", seq)

    radar_pkl = os.path.join(RADAR_DIR, f"{seq}_read.pkl")
    lidar_csv = glob.glob(os.path.join(LIDAR_DIR, f"{seq}_fwd.*"))[0]
    align_pkl = os.path.join(ALIGN_DIR, f"{seq}_time_alignment.pkl")

    # ---------- Radar ----------
    with open(radar_pkl, "rb") as f:
        radar = pickle.load(f)

    radar_frames = radar["frames"]

    # ---------- 时间对齐 ----------
    with open(align_pkl, "rb") as f:
        align = pickle.load(f)

    start_idx = align["radar_start_idx"]
    end_idx   = align["radar_end_idx"]

    print("Radar usable frames:", start_idx, "→", end_idx)

    # ---------- LiDAR ----------
    lidar_df = pd.read_csv(lidar_csv, low_memory=False)
    lidar_xyz = lidar_df[["X", "Y", "Z"]].to_numpy(dtype=np.float32)

    lidar_occ, lidar_height = lidar_to_bev_occ_height(lidar_xyz)

    # ---------- 保存 ----------
    save_seq_dir = os.path.join(SAVE_DIR, seq)
    os.makedirs(save_seq_dir, exist_ok=True)

    sample_idx = 0

    for i in range(start_idx, end_idx - T + 1):

        radar_stack = np.stack([
            radar_points_to_bev(radar_frames[i + k])
            for k in range(T)
        ], axis=0)

        with open(
            os.path.join(save_seq_dir, f"{sample_idx:06d}.pkl"),
            "wb"
        ) as f:
            pickle.dump(
                {
                    "radar": radar_stack,
                    "lidar_occ": lidar_occ,
                    "lidar_height": lidar_height
                },
                f
            )

        sample_idx += 1

    print("✅ 样本生成完成，总数:", sample_idx)


# =========================
# 入口
# =========================
if __name__ == "__main__":

    radar_files = glob.glob(os.path.join(RADAR_DIR, "*_read.pkl"))

    for f in radar_files:
        seq = os.path.basename(f).replace("_read.pkl", "")
        process_sequence(seq)
