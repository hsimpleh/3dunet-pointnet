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
SAVE_DIR  = "D:/毕业设计/RedarHD-3D/create_dataset/time_alignment"

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================
# 安全解析 LiDAR 时间戳
# =========================================================
def safe_parse_timestamp(ts):
    if ts is None:
        return None
    s = str(ts).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


# =========================================================
def process_sequence(seq):

    print("\n==============================")
    print(f"开始处理序列: {seq}")

    radar_pkl = os.path.join(RADAR_DIR, f"{seq}_read.pkl")
    lidar_csv = glob.glob(os.path.join(LIDAR_DIR, f"{seq}_fwd.*"))[0]

    # -----------------------------
    # 1. Radar
    # -----------------------------
    with open(radar_pkl, "rb") as f:
        radar = pickle.load(f)

    radar_ts = radar["timestamps"].astype(np.int64)
    num_frames = len(radar_ts)

    radar_start_ts = radar_ts[0]
    radar_end_ts   = radar_ts[-1]

    print(f"雷达总帧数: {num_frames}")
    print(f"雷达时间跨度(us): {radar_end_ts - radar_start_ts}")

    # -----------------------------
    # 2. LiDAR
    # -----------------------------
    lidar_df = pd.read_csv(lidar_csv, dtype=str, low_memory=False)
    lidar_ts_raw = lidar_df.iloc[:, 6].values

    lidar_ts = []
    for x in lidar_ts_raw:
        t = safe_parse_timestamp(x)
        if t is None or t < 1e10:
            continue
        lidar_ts.append(t)

    lidar_ts = np.array(lidar_ts, dtype=np.int64)

    if lidar_ts.max() < 1e13:
        lidar_ts *= 1_000  # ms → us

    lidar_start = lidar_ts.min()
    lidar_end   = lidar_ts.max()

    print(f"LiDAR 时间跨度(us): {lidar_end - lidar_start}")

    # -----------------------------
    # 3. 归一化到相对时间轴
    # -----------------------------
    radar_rel = radar_ts - radar_start_ts
    lidar_rel_start = 0
    lidar_rel_end   = lidar_end - lidar_start

    # -----------------------------
    # 4. 按时间比例对齐
    # -----------------------------
    ratio_start = lidar_rel_start / lidar_rel_end
    ratio_end   = lidar_rel_end   / lidar_rel_end  # =1

    start_idx = int(ratio_start * num_frames)
    end_idx   = int(ratio_end   * num_frames) - 1

    start_idx = max(0, start_idx)
    end_idx   = min(num_frames - 1, end_idx)

    # -----------------------------
    # 5. 保存结果
    # -----------------------------
    alignment = {
        "seq": seq,
        "radar_start_idx": start_idx,
        "radar_end_idx": end_idx,
        "radar_relative_timestamps": radar_rel[start_idx:end_idx + 1],
        "lidar_relative_time_range": [0, lidar_rel_end]
    }

    save_path = os.path.join(SAVE_DIR, f"{seq}_time_alignment.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(alignment, f)

    print(f"✅ 相对时间对齐完成")
    print(f"雷达可用帧数: {end_idx - start_idx + 1}")


# =========================================================
if __name__ == "__main__":

    radar_files = glob.glob(os.path.join(RADAR_DIR, "*_read.pkl"))

    for f in radar_files:
        seq = os.path.basename(f).replace("_read.pkl", "")
        process_sequence(seq)
