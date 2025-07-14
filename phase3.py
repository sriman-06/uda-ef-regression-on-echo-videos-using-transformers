import torch
import pandas as pd
from torch.utils.data import DataLoader
from phase import MultiFrameDeiT, VideoDataset, train_uda, validate, train_transform, val_transform, get_sampler

# ========= Paths =========
PHASE1_WEIGHTS = "phase1_model.pt"
SOURCE_CSV = r"C:/Users/Admin/Downloads/EchoNet-Dynamic/EchoNet-Dynamic/FileList.csv"
SOURCE_TENSORS = r"C:/Users/Admin/Downloads/EchoNet-Dynamic/processed_videos"

TARGET_CSV = r"C:/Users/Admin/Downloads/EchoNet-LVH/echonetlvh/EchoNet-LVH/lvh_pseudo_labels.csv"
TARGET_TENSORS = r"D:/preprocessed_lvh_tensors"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8

# ========== Load CSVs ==========
src_df = pd.read_csv(SOURCE_CSV)
src_df['label'] = src_df['EF']
src_df['Age'] = src_df.get('Age', pd.Series([65]*len(src_df)))
src_df['Sex'] = src_df.get('Sex', pd.Series([1]*len(src_df)))
src_df['BP'] = src_df.get('BP', pd.Series([120]*len(src_df)))

tgt_df = pd.read_csv(TARGET_CSV).rename(columns={'HashedFileName': 'FileName'})
tgt_df['Age'] = tgt_df.get('Age', pd.Series([65]*len(tgt_df)))
tgt_df['Sex'] = tgt_df.get('Sex', pd.Series([1]*len(tgt_df)))
tgt_df['BP'] = tgt_df.get('BP', pd.Series([120]*len(tgt_df)))

# ========== Prepare DataLoaders ==========
# Source loaders
src_train = src_df[src_df['Split'] == 'TRAIN']
src_dataset = VideoDataset(src_train, SOURCE_TENSORS, train_transform)
src_loader = DataLoader(src_dataset, batch_size=BATCH_SIZE, sampler=get_sampler(src_train), num_workers=0)

# Target loaders
tgt_train = tgt_df[tgt_df['split'] == 'train']
tgt_val = tgt_df[tgt_df['split'] == 'val']

tgt_train_dataset = VideoDataset(tgt_train, TARGET_TENSORS, train_transform)
tgt_val_dataset = VideoDataset(tgt_val, TARGET_TENSORS, val_transform)

tgt_loader = DataLoader(tgt_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
tgt_val_loader = DataLoader(tgt_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ========== Load Phase 1 Weights ==========
model = MultiFrameDeiT()
model.load_state_dict(torch.load(PHASE1_WEIGHTS, map_location=DEVICE))
model.to(DEVICE)

# ========== Phase 1 Baseline on LVH ==========
print("\n📊 Evaluating Phase 1 model on LVH before UDA...")
baseline_mae = validate(
    model, tgt_val_loader,
    plot_title="Phase 1 on LVH (Baseline)",
    save_plot="phase1_on_lvh.png"
)
print(f"🎯 Phase 1 Baseline MAE on LVH: {baseline_mae:.2f}")

# ========== Phase 3: Domain-Adaptive UDA ==========
print("\n🚀 Starting Phase 3 UDA Training...")
train_uda(model, src_loader, tgt_loader, tgt_val_loader)

# ========== Final Comparison ==========
try:
    phase3_log = pd.read_csv("uda_metrics_log.csv")
    final_uda_mae = phase3_log["MAE"].min()
    print(f"\n📊 Final UDA MAE: {final_uda_mae:.2f}")
    if final_uda_mae < baseline_mae:
        print("✅ UDA improved performance over baseline!")
    else:
        print("⚠️ UDA did not beat baseline. Try tuning λ_mmd or pseudo-label strategy.")
except Exception as e:
    print(f"❌ Final comparison failed: {e}")
