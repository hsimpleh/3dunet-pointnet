"""
本脚本用于将毫米波雷达原始 txt 文件解析并保存为 pkl 格式
支持对整个文件夹内的所有 txt 文件进行批量处理
解析后的 pkl 文件用于后续时间同步、数据集构建与网络训练
"""

import os
import pickle
import numpy as np


def safe_parse_timestamp_ms(ts):
    """
    安全解析时间戳（毫秒）
    统一返回 int64，避免 float 精度丢失
    """
    s = str(ts).strip()

    if s == "" or s.lower() == "nan":
        return None

    # 科学计数法或小数，统一转 float 再转 int
    if "e" in s or "." in s:
        return int(float(s))

    return int(s)


def parse_radar_txt(txt_path):
    """
    解析单个毫米波雷达 txt 文件

    参数：
        txt_path: 雷达 txt 文件路径

    返回：
        radar_data: dict
            - timestamps: np.ndarray (int64, us)
            - frames: List[np.ndarray], 每帧形状为 (Ni, 6)
            - num_frames: int
    """

    frames = []            # 每一帧点云 (Ni, 6)
    timestamps_us = []     # 每一帧时间戳（微秒）

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    idx = 0
    total_lines = len(lines)

    while idx < total_lines:
        # =========================
        # 1. 读取帧头
        # =========================
        header = lines[idx].strip()
        idx += 1

        if header == "":
            continue

        header_parts = header.split(',')
        if len(header_parts) < 2:
            print(f"⚠️ 帧头格式异常，跳过：{header}")
            continue

        timestamp_ms = safe_parse_timestamp_ms(header_parts[0])
        num_points = int(float(header_parts[1]))

        if timestamp_ms is None or num_points <= 0:
            print("⚠️ 时间戳或点数异常，跳过该帧")
            idx += num_points
            continue

        # =========================
        # 2. 读取点云
        # =========================
        points = []
        for _ in range(num_points):
            if idx >= total_lines:
                break

            line = lines[idx].strip()
            idx += 1

            if line == "":
                continue

            vals = line.split(',')
            if len(vals) < 6:
                continue

            point = list(map(float, vals[:6]))
            points.append(point)

        if len(points) == 0:
            print("⚠️ 空点云帧，跳过")
            continue

        frames.append(np.asarray(points, dtype=np.float32))
        timestamps_us.append(timestamp_ms * 1000)  # ms → us

    radar_data = {
        "timestamps": np.asarray(timestamps_us, dtype=np.int64),
        "frames": frames,
        "num_frames": len(frames)
    }

    return radar_data


def save_radar_pkl(txt_path, save_dir):
    """
    将单个毫米波雷达 txt 文件解析并保存为 pkl
    """

    radar_data = parse_radar_txt(txt_path)

    os.makedirs(save_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(txt_path))[0]
    save_path = os.path.join(save_dir, base_name + "_read.pkl")

    with open(save_path, "wb") as f:
        pickle.dump(radar_data, f)

    print("====================================")
    print(f"📄 TXT 文件：{txt_path}")
    print(f"✅ 已生成 PKL：{save_path}")
    print(f"总帧数：{radar_data['num_frames']}")
    if radar_data["num_frames"] > 0:
        print(f"第一帧点云形状：{radar_data['frames'][0].shape}")
    print("====================================")


def process_radar_txt_folder(txt_dir, save_dir):
    """
    处理整个文件夹中的所有 radar txt 文件
    """

    if not os.path.isdir(txt_dir):
        raise ValueError(f"❌ 输入路径不是文件夹：{txt_dir}")

    txt_files = sorted([
        os.path.join(txt_dir, f)
        for f in os.listdir(txt_dir)
        if f.lower().endswith(".txt")
    ])

    if len(txt_files) == 0:
        print("⚠️ 文件夹中未找到 txt 文件")
        return

    print(f"📂 共检测到 {len(txt_files)} 个雷达 txt 文件")

    for txt_path in txt_files:
        try:
            save_radar_pkl(txt_path, save_dir)
        except Exception as e:
            print(f"❌ 处理失败：{txt_path}")
            print(e)


if __name__ == "__main__":

    # ===== 修改为你自己的路径 =====
    radar_txt_dir = "D:/毕业设计/RedarHD-3D/data/radar"
    radar_pkl_save_dir = "D:/毕业设计/RedarHD-3D/create_dataset/radar_dataset"

    process_radar_txt_folder(radar_txt_dir, radar_pkl_save_dir)
