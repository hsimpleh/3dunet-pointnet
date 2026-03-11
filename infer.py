import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Subset

# ================= 项目路径 =================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from train_test_utils.dataloader import RadarLidarDataset
from models.radarhd_unet3d import Radar3DUNet


# ================= 配置 =================

DATASET_DIR = "/home/csw/Radar3DUNet/RedarHD-3D/create_dataset/dataset_unet"
SEQ_LIST = [f"S{i}" for i in range(1,4)]

MODEL_A = "/home/csw/Radar3DUNet/RedarHD-3D/checkpoints_v4/model_epoch19.pth"
MODEL_B = "/home/csw/Radar3DUNet/RedarHD-3D/checkpoints_v4-10/model_epoch10.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 推理数据比例
INFER_RATIO = 0.1

# 阈值扫描
THRESHOLDS = np.linspace(0.05,0.95,19)

SAVE_DIR = "./infer_results"
os.makedirs(SAVE_DIR,exist_ok=True)


# ================= 加载模型 =================

def load_model(model,path):

    ckpt = torch.load(path,map_location=DEVICE)

    state = ckpt["model"] if "model" in ckpt else ckpt

    new_state = {}

    for k,v in state.items():

        if k.startswith("module."):

            k = k[7:]

        new_state[k] = v

    model.load_state_dict(new_state)

    return model


# ================= 指标计算 =================

def compute_iou(pred,gt,t):

    pred = (pred>t).float()

    inter = (pred*gt).sum()
    union = pred.sum()+gt.sum()-inter

    return (inter/(union+1e-6)).item()


def compute_precision(pred,gt,t):

    pred = (pred>t).float()

    tp = (pred*gt).sum()
    fp = (pred*(1-gt)).sum()

    return (tp/(tp+fp+1e-6)).item()


def compute_recall(pred,gt,t):

    pred = (pred>t).float()

    tp = (pred*gt).sum()
    fn = ((1-pred)*gt).sum()

    return (tp/(tp+fn+1e-6)).item()


def compute_mse(pred,gt):

    return torch.mean((pred-gt)**2).item()


# ================= 可视化 =================

def visualize(idx,occ_gt,occ_pred,hgt_gt,hgt_pred):

    occ_gt = occ_gt.squeeze().cpu()
    occ_pred = occ_pred.squeeze().cpu()

    hgt_gt = hgt_gt.squeeze().cpu()
    hgt_pred = hgt_pred.squeeze().cpu()

    error = torch.abs(occ_pred-occ_gt)

    fig = plt.figure(figsize=(10,8))

    plt.subplot(2,2,1)
    plt.title("GT Occupancy")
    plt.imshow(occ_gt,cmap="gray")

    plt.subplot(2,2,2)
    plt.title("Pred Occupancy")
    plt.imshow(occ_pred,cmap="gray")

    plt.subplot(2,2,3)
    plt.title("GT Height")
    plt.imshow(hgt_gt,cmap="jet")

    plt.subplot(2,2,4)
    plt.title("Error Map")
    plt.imshow(error,cmap="hot")

    plt.tight_layout()

    plt.savefig(os.path.join(SAVE_DIR,f"vis_{idx}.png"))

    plt.close()


# ================= 主函数 =================

def main():

    print("🚀 Device:",DEVICE)

    dataset = RadarLidarDataset(DATASET_DIR,SEQ_LIST)

    total = len(dataset)

    print(f"📦 TRAIN samples: {total}")

    # ===== 数据比例 =====
    use_num = int(total*INFER_RATIO)

    indices = np.random.choice(total,use_num,replace=False)

    dataset = Subset(dataset,indices)

    print(f"📦 Samples used: {use_num}")

    # ===== 模型 =====

    modelA = Radar3DUNet(in_channels=1,base_ch=64).to(DEVICE)
    modelB = Radar3DUNet(in_channels=1,base_ch=64).to(DEVICE)

    modelA = load_model(modelA,MODEL_A)
    modelB = load_model(modelB,MODEL_B)

    modelA.eval()
    modelB.eval()

    predsA=[]
    predsB=[]
    gts=[]

    heightsA=[]
    heightsB=[]
    heightsGT=[]

    # ================= 推理 =================

    with torch.no_grad():

        for idx in tqdm(range(len(dataset))):

            sample = dataset[idx]

            radar = sample["radar"].unsqueeze(0).to(DEVICE)

            occ_gt = sample["occ"].to(DEVICE)
            hgt_gt = sample["height"].to(DEVICE)

            occA,hgtA = modelA(radar)
            occB,hgtB = modelB(radar)

            occA = torch.sigmoid(occA)[0,0]
            occB = torch.sigmoid(occB)[0,0]

            hgtA = hgtA[0,0]
            hgtB = hgtB[0,0]

            predsA.append(occA.cpu())
            predsB.append(occB.cpu())

            gts.append(occ_gt.cpu())

            heightsA.append(hgtA.cpu())
            heightsB.append(hgtB.cpu())

            heightsGT.append(hgt_gt.cpu())

            if idx < 10:

                visualize(idx,occ_gt,occA,hgt_gt,hgtA)

    # ================= Threshold 搜索 =================

    iouA=[]
    iouB=[]

    precisionA=[]
    recallA=[]

    for t in THRESHOLDS:

        iou_tmpA=[]
        iou_tmpB=[]

        p_tmp=[]
        r_tmp=[]

        for pA,pB,g in zip(predsA,predsB,gts):

            iou_tmpA.append(compute_iou(pA,g,t))
            iou_tmpB.append(compute_iou(pB,g,t))

            p_tmp.append(compute_precision(pA,g,t))
            r_tmp.append(compute_recall(pA,g,t))

        iouA.append(np.mean(iou_tmpA))
        iouB.append(np.mean(iou_tmpB))

        precisionA.append(np.mean(p_tmp))
        recallA.append(np.mean(r_tmp))

    # ================= Best IoU =================

    best_idxA = np.argmax(iouA)
    best_idxB = np.argmax(iouB)

    best_tA = THRESHOLDS[best_idxA]
    best_tB = THRESHOLDS[best_idxB]

    best_iouA = iouA[best_idxA]
    best_iouB = iouB[best_idxB]

    print("\n===== BEST RESULT =====")

    print(f"Model3 -> Best IoU: {best_iouA:.4f} | Threshold: {best_tA}")
    print(f"Model10 -> Best IoU: {best_iouB:.4f} | Threshold: {best_tB}")

    # ================= Average IoU =================

    print("\n===== AVERAGE IoU =====")

    print("Model3 Avg IoU:",np.mean(iouA))
    print("Model10 Avg IoU:",np.mean(iouB))

    # ================= Height Error =================

    mseA=[compute_mse(p,g) for p,g in zip(heightsA,heightsGT)]
    mseB=[compute_mse(p,g) for p,g in zip(heightsB,heightsGT)]

    print("\n===== HEIGHT ERROR =====")

    print("ModelA MSE:",np.mean(mseA))
    print("ModelB MSE:",np.mean(mseB))

    # ================= IoU Curve =================

    plt.figure()

    plt.plot(THRESHOLDS,iouA,label="Model3")
    plt.plot(THRESHOLDS,iouB,label="Model10")

    plt.xlabel("Threshold")
    plt.ylabel("IoU")

    plt.legend()

    plt.savefig(os.path.join(SAVE_DIR,"iou_curve.png"))

    # ================= PR Curve =================

    plt.figure()

    plt.plot(recallA,precisionA)

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.savefig(os.path.join(SAVE_DIR,"pr_curve.png"))

    print("\n📊 Evaluation finished.")


if __name__=="__main__":

    main()