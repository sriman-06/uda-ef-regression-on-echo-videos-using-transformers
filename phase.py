# ✅ Full Enhanced Phase 1 + Phase 3 EF Regression Pipeline (Clean & Corrected)

import os
import time
import torch
import torch.nn as nn
import timm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import mean_absolute_error, r2_score
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import copy
import math

# ================= Config =================
NUM_FRAMES = 8
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ======== Utility Functions ========
def get_sampler(df):
    bins = pd.cut(df['label'], bins=[0, 30, 50, 70, 100], labels=False)
    class_counts = bins.value_counts().sort_index().values
    weights = 1. / class_counts[bins]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def cosine_rampdown(current, rampdown_length):
    return float(.5 * (math.cos(math.pi * current / rampdown_length) + 1))

def sharpen(prob, T=0.5):
    prob = torch.clamp(prob, 1e-6, 1 - 1e-6)
    sharp = prob ** (1 / T)
    return sharp / sharp.sum(dim=-1, keepdim=True)

def mixup(x1, y1, x2, y2, alpha=0.4):
    l = np.random.beta(alpha, alpha)
    x_mix = l * x1 + (1 - l) * x2
    y_mix = l * y1 + (1 - l) * y2
    return x_mix, y_mix

# ================ Dataset =================
class VideoDataset(Dataset):
    def __init__(self, df, video_dir, transform):
        self.df = df.reset_index(drop=True)
        self.video_dir = video_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['FileName']  # unique identifier
        tensor_path = os.path.join(self.video_dir, fname + '.pt')
        video_tensor = torch.load(tensor_path).to(memory_format=torch.channels_last)
        label = torch.tensor(row['label'], dtype=torch.float32)
        label += torch.empty_like(label).uniform_(-1.0, 1.0)  # Label smoothing
        meta = torch.tensor([row['Age'], row['Sex'], row['BP']], dtype=torch.float32)
        video_tensor = torch.stack([
            self.transform(frame.permute(1, 2, 0).numpy()) for frame in video_tensor
        ])
        return video_tensor, meta, label, fname  # ✅ include filename

# ================ Model ===================
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=NUM_FRAMES):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )

    def forward(self, x):
        weights = self.attn(x)
        weights = torch.softmax(weights, dim=1)
        return (x * weights).sum(dim=1)

class MultiFrameDeiT(nn.Module):
    def __init__(self, metadata_dim=3, num_classes=1):
        super().__init__()
        self.base = timm.create_model("deit_base_patch16_224", pretrained=True, drop_path_rate=0.2)
        self.base.head = nn.Identity()
        self.pos_enc = PositionalEncoding(768)
        self.meta_net = nn.Sequential(
            nn.Linear(metadata_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.Sigmoid()
        )
        self.temporal_pool = AttentionPool(768)
        self.classifier = nn.Sequential(
            nn.Linear(768 + 32, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, video, meta):
        b, t, c, h, w = video.shape
        video = video.view(b * t, c, h, w).contiguous(memory_format=torch.channels_last)
        feat = self.base(video).view(b, t, -1)
        feat = self.pos_enc(feat)
        pooled = self.temporal_pool(feat)
        meta_feat = self.meta_net(meta)
        combined = torch.cat([pooled, meta_feat], dim=1)
        out = self.classifier(combined).squeeze(1)
        return 100 * torch.sigmoid(out), combined  # scaled output

# ============ MK-MMD Loss ============
class MKMMDLoss(nn.Module):
    def __init__(self, kernels=[0.5, 1.0]):
        super().__init__()
        self.kernels = kernels

    def gaussian_kernel(self, x, y):
        delta = x.unsqueeze(1) - y.unsqueeze(0)
        l2 = delta.pow(2).sum(2)
        return sum(torch.exp(-l2 / (2 * k)) for k in self.kernels)

    def forward(self, source, target):
        return (
            self.gaussian_kernel(source, source).mean() +
            self.gaussian_kernel(target, target).mean() -
            2 * self.gaussian_kernel(source, target).mean()
        )

# === Supervised Training (Corrected) ===
def train_supervised(model, loader, val_loader, phase_name="Phase1", save_name="model.pt"):
    if any(p.dim() == 4 for p in model.parameters()):
        model.to(DEVICE, memory_format=torch.channels_last)
    else:
        model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    criterion = nn.SmoothL1Loss(beta=1.0)
    best_mae = float('inf')
    ema_model = copy.deepcopy(model)
    decay = 0.99
    log_file = "metrics_log.csv"
    with open(log_file, "w") as f:
        f.write("Epoch,Phase,MAE,R2\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for video, meta, label, _ in tqdm(loader):
            video, meta, label = video.to(DEVICE), meta.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            preds, _ = model(video, meta)
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)

        print(f"[{phase_name} Epoch {epoch+1}] ✅ Loss: {total_loss:.4f}")
        mae = validate(ema_model, val_loader, plot_title=f"{phase_name} Validation", save_plot=f"{phase_name.lower()}_val_epoch{epoch+1}.png", log_file=log_file, epoch=epoch+1, phase=phase_name)
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), save_name)
            print("✅ Best model saved!")

    # Plot MAE trend
    df = pd.read_csv(log_file)
    plt.figure(figsize=(8, 4))
    plt.plot(df["Epoch"], df["MAE"], marker="o", label="MAE")
    plt.title("MAE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.legend()
    plt.savefig("mae_over_epochs.png")
    print("📈 MAE plot saved.")
    print(f"🎯 Best MAE during {phase_name}: {best_mae:.2f}")
    
def tta_predict(model, video, meta, tta_times=5):
    preds = []
    for _ in range(tta_times):
        noisy = video.clone()
        if torch.rand(1).item() > 0.5:
            noisy = torch.flip(noisy, dims=[-1])  # horizontal flip
        if torch.rand(1).item() > 0.5:
            noisy = torch.rot90(noisy, k=1, dims=[-2, -1])  # 90-degree rotation
        pred, _ = model(noisy.to(DEVICE), meta.to(DEVICE))
        preds.append(pred.cpu().numpy())
    return np.mean(preds, axis=0)
    

# ============ Validation ============
def validate(model, loader, plot_title="EF vs Predicted EF", save_plot=None, log_file=None, epoch=None, phase=""):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for video, meta, label, _ in loader:

            video, meta = video.to(DEVICE), meta.to(DEVICE)
            tta_out = tta_predict(model, video, meta)
            preds.extend(tta_out)
            targets.extend(label.numpy())
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    print(f"🔍 MAE: {mae:.2f}, R²: {r2:.2f}")

    if save_plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(targets, preds, c='blue', alpha=0.5, edgecolors='k')
        plt.plot([0, 100], [0, 100], 'r--', label="Ideal (y = x)")
        plt.xlabel("True EF (%)")
        plt.ylabel("Predicted EF (%)")
        plt.title(plot_title)
        plt.grid(True)
        plt.legend()
        plt.savefig(save_plot)
        print(f"📊 Plot saved to: {save_plot}")

    if log_file and epoch is not None:
        with open(log_file, "a") as f:
            f.write(f"{epoch},{phase},{mae:.4f},{r2:.4f}\n")

    return mae
# ============ UDA Training ============
def train_uda(
    model,
    source_loader,
    target_loader,
    val_loader,
    lambda_mmd=0.05,
    lambda_pseudo=0.1,
    lambda_entropy=0.01,
    conf_thresh=0.95,
    feature_noise=0.005,
    epochs=EPOCHS,
    phase_name="Phase3"
):
    if any(p.dim() == 4 for p in model.parameters()):
        model.to(DEVICE, memory_format=torch.channels_last)
    else:
        model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    scaler = GradScaler()
    criterion = nn.MSELoss()
    mmd_loss = MKMMDLoss()
    best_mae = float("inf")
    ema_model = copy.deepcopy(model)
    decay = 0.99
    log_file = "uda_metrics_log.csv"

    with open(log_file, "w") as f:
        f.write("Epoch,Phase,MAE,R2\n")

    ema_pseudo_dict = {}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pseudo_weight = lambda_pseudo * cosine_rampdown(epoch, epochs)

        for (src_vid, src_meta, src_label, _), (tgt_vid, tgt_meta, _, tgt_fnames) in tqdm(
            zip(cycle(source_loader), target_loader),
            total=min(len(source_loader), len(target_loader))
        ):
            src_vid, src_meta, src_label = src_vid.to(DEVICE), src_meta.to(DEVICE), src_label.to(DEVICE)
            tgt_vid, tgt_meta = tgt_vid.to(DEVICE), tgt_meta.to(DEVICE)
            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                src_pred, src_feat = model(src_vid, src_meta)
                tgt_pred, tgt_feat = model(tgt_vid, tgt_meta)

                if feature_noise > 0:
                    src_feat = src_feat + torch.randn_like(src_feat) * feature_noise
                    tgt_feat = tgt_feat + torch.randn_like(tgt_feat) * feature_noise

                loss_reg = criterion(src_pred, src_label)
                loss = loss_reg

                if epoch >= 2:
                    loss += lambda_mmd * mmd_loss(src_feat, tgt_feat)

                    p = torch.sigmoid(tgt_pred)
                    entropy = - (p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6))
                    loss += lambda_entropy * entropy.mean()

                if epoch >= 5:
                    with torch.no_grad():
                        ema_out, _ = ema_model(tgt_vid, tgt_meta)
                        pseudo_probs = torch.sigmoid(ema_out.unsqueeze(1))
                        sharp_pseudo = sharpen(pseudo_probs)
                        for fname, pseudo_ef in zip(tgt_fnames, ema_out):
                            ema_pseudo_dict[fname] = pseudo_ef.item()

                    conf_mask = ((pseudo_probs > conf_thresh) | (pseudo_probs < 1 - conf_thresh)).squeeze()
                    if conf_mask.sum() > 0:
                        pseudo_labels = ema_out[conf_mask].detach()
                        pseudo_vid = tgt_vid[conf_mask]
                        pseudo_meta = tgt_meta[conf_mask]

                        mix_size = min(src_vid.size(0), pseudo_vid.size(0))
                        if mix_size > 0:
                            # ✅ Step 1: Match number of frames (T)
                            min_T = min(src_vid.shape[1], pseudo_vid.shape[1])
                            src_vid = src_vid[:, :min_T]
                            pseudo_vid = pseudo_vid[:, :min_T]

                            # ✅ Step 2: Perform mixup on video + labels
                            mixed_video, mixed_labels = mixup(
                                src_vid[:mix_size], src_label[:mix_size].unsqueeze(1),
                                pseudo_vid[:mix_size], pseudo_labels[:mix_size].unsqueeze(1)
                            )

                            # ✅ Step 3: Mix metadata the same way (dummy target for shape)
                            mixed_meta = mixup(
                                src_meta[:mix_size], torch.zeros_like(src_meta[:mix_size]),
                                pseudo_meta[:mix_size], torch.zeros_like(pseudo_meta[:mix_size])
                            )[0]  # take only the mixed features

                            # ✅ Step 4: Ensure batch match
                            assert mixed_video.shape[0] == mixed_meta.shape[0], "Batch size mismatch before model"

                            # ✅ Step 5: Forward with mixed inputs
                            mixed_preds, _ = model(mixed_video, mixed_meta)
                            loss += pseudo_weight * criterion(mixed_preds.squeeze(), mixed_labels.squeeze())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data = ema_p.data * decay + p.data * (1 - decay)

        print(f"[UDA Epoch {epoch+1}] ✅ Total Loss: {total_loss:.4f}")
        scheduler.step()

        mae = validate(
            ema_model, val_loader,
            plot_title=f"UDA Validation Epoch {epoch+1}",
            save_plot=f"uda_epoch{epoch+1}.png",
            log_file=log_file,
            epoch=epoch+1,
            phase=phase_name,
        )

        if mae < best_mae:
            best_mae = mae
            torch.save(ema_model.state_dict(), "best_uda_regression.pt")
            print("✅ Best UDA model saved!")

    if ema_pseudo_dict:
        pseudo_df = pd.DataFrame([{"FileName": k, "PseudoEF": v} for k, v in ema_pseudo_dict.items()])
        pseudo_df.to_csv("target_confident_pseudo_labels.csv", index=False)
        print("📝 Saved confident pseudo-labels to: target_confident_pseudo_labels.csv")
    else:
        print("⚠️ No confident pseudo-labels were collected to save.")

    df = pd.read_csv(log_file)
    plt.figure(figsize=(8, 4))
    plt.plot(df["Epoch"], df["MAE"], marker="o", label="UDA MAE")
    plt.title("UDA MAE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.legend()
    plt.savefig("uda_mae_over_epochs.png")
    print("📈 UDA MAE plot saved.")
    print(f"🎯 Best MAE during {phase_name}: {best_mae:.2f}")





if __name__ == "__main__":
    print("✅ Starting EF Regression with Enhancements")

    # 🔁 Move target DF loading outside so it's accessible to both phases
    tgt_df = pd.read_csv(r"C:/Users/Admin/Downloads/EchoNet-LVH/echonetlvh/EchoNet-LVH/lvh_pseudo_labels.csv")
    tgt_df = tgt_df.rename(columns={'HashedFileName': 'FileName'})
    for df in [tgt_df]:
        df['Age'] = df.get('Age', pd.Series([65]*len(df)))
        df['Sex'] = df.get('Sex', pd.Series([1]*len(df)))
        df['BP'] = df.get('BP', pd.Series([120]*len(df)))

    try:
        # Phase 1: Supervised EF Regression
        src_df = pd.read_csv(r"C:/Users/Admin/Downloads/EchoNet-Dynamic/EchoNet-Dynamic/FileList.csv")
        src_df['label'] = src_df['EF']
        for df in [src_df]:
            df['Age'] = df.get('Age', pd.Series([65]*len(df)))
            df['Sex'] = df.get('Sex', pd.Series([1]*len(df)))
            df['BP'] = df.get('BP', pd.Series([120]*len(df)))

        src_train = src_df[src_df['Split'] == 'TRAIN']
        src_val = src_df[src_df['Split'] == 'VAL']

        src_dataset = VideoDataset(src_train, r"C:/Users/Admin/Downloads/EchoNet-Dynamic/processed_videos", train_transform)
        src_val_dataset = VideoDataset(src_val, r"C:/Users/Admin/Downloads/EchoNet-Dynamic/processed_videos", val_transform)

        src_loader = DataLoader(src_dataset, batch_size=BATCH_SIZE, sampler=get_sampler(src_train), num_workers=0)
        src_val_loader = DataLoader(src_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = MultiFrameDeiT()
        train_supervised(model, src_loader, src_val_loader, phase_name="Phase1", save_name="phase1_model.pt")
        print("✅ Phase 1 completed successfully!\n")

        # 📊 Evaluate Phase 1 model on LVH before UDA (DA Baseline)
        print("📊 Evaluating Phase 1 model on LVH before UDA...")
        tgt_val = tgt_df[tgt_df['split'] == 'val']
        tgt_val_dataset = VideoDataset(tgt_val, r"D:/preprocessed_lvh_tensors", val_transform)
        tgt_val_loader = DataLoader(tgt_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model.load_state_dict(torch.load("phase1_model.pt"))
        baseline_mae = validate(
            model, tgt_val_loader,
            plot_title="Phase 1 on LVH (Baseline)",
            save_plot="phase1_on_lvh.png"
        )
        print(f"🎯 Baseline MAE on LVH before UDA: {baseline_mae:.2f}")

    except Exception as e:
        print(f"❌ Phase 1 or baseline eval failed: {e}\n")

    try:
        # Phase 3: Domain-Adaptive EF Regression
        tgt_train = tgt_df[tgt_df['split'] == 'train']
        tgt_dataset = VideoDataset(tgt_train, r"D:/preprocessed_lvh_tensors", train_transform)
        tgt_loader = DataLoader(tgt_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        # ✅ Load Phase 1 pretrained weights again for UDA
        model.load_state_dict(torch.load("phase1_model.pt"))

        # ✅ Start UDA training
        train_uda(model, src_loader, tgt_loader, tgt_val_loader)

        print("✅ Phase 3 completed successfully!")

    except Exception as e:
        print(f"❌ Phase 3 failed: {e}\n")

    # ✅ Final MAE comparison (Phase 1 vs Phase 3)
    try:
        phase1_log = pd.read_csv("metrics_log.csv")
        phase3_log = pd.read_csv("uda_metrics_log.csv")

        baseline_mae = phase1_log["MAE"].min()
        final_uda_mae = phase3_log["MAE"].min()

        print(f"\n📊 Baseline (Phase 1) MAE: {baseline_mae:.2f}")
        print(f"📊 Final (Phase 3) MAE: {final_uda_mae:.2f}")

        if final_uda_mae < baseline_mae:
            print("✅ Domain Adaptation successful! Phase 3 outperformed Phase 1.")
        else:
            print("⚠️ DA failed to beat Phase 1. Try tuning λ_mmd, pseudo-labels, or add TTA.")
    except Exception as e:
        print(f"⚠️ Final comparison failed: {e}") 