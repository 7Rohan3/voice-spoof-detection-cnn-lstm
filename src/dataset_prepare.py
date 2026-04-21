import os
import shutil
import random

DATASET_ROOT = "dataset_full"

PROTOCOL_FILE = os.path.join(
    DATASET_ROOT,
    "ASVspoof2017_train.trn.txt"
)

TRAIN_WAV_FOLDER = os.path.join(
    DATASET_ROOT,
    "ASVspoof2017_train"
)

TARGET_REAL = "dataset/real"
TARGET_SPOOF = "dataset/spoof"

os.makedirs(TARGET_REAL, exist_ok=True)
os.makedirs(TARGET_SPOOF, exist_ok=True)

real_files = []
spoof_files = []

with open(PROTOCOL_FILE, "r") as f:

    for line in f:

        parts = line.strip().split()

        filename = parts[0]   # <-- FIXED
        label = parts[1].lower()  # <-- FIXED

        wav_path = os.path.join(
            TRAIN_WAV_FOLDER,
            filename
        )

        if label == "genuine":
            real_files.append(wav_path)

        else:
            spoof_files.append(wav_path)

print("Total real files:", len(real_files))
print("Total spoof files:", len(spoof_files))

# Select safe subset
num_real = min(500, len(real_files))
num_spoof = min(500, len(spoof_files))

real_selected = random.sample(real_files, num_real)
spoof_selected = random.sample(spoof_files, num_spoof)

# Copy real
for file_path in real_selected:

    shutil.copy(
        file_path,
        os.path.join(
            TARGET_REAL,
            os.path.basename(file_path)
        )
    )

# Copy spoof
for file_path in spoof_selected:

    shutil.copy(
        file_path,
        os.path.join(
            TARGET_SPOOF,
            os.path.basename(file_path)
        )
    )

print("Dataset preparation complete.")