import os
import numpy as np
import librosa
import pickle

DATASET_PATH = "dataset"

MAX_LEN = 100
N_MFCC = 40

def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        duration=2
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc = mfcc.T

    # Padding
    if mfcc.shape[0] < MAX_LEN:

        pad_width = MAX_LEN - mfcc.shape[0]

        pad = np.zeros(
            (pad_width, mfcc.shape[1])
        )

        mfcc = np.vstack((mfcc, pad))

    else:

        mfcc = mfcc[:MAX_LEN]

    return mfcc


X = []
y = []

labels = ["real", "spoof"]

for label in labels:

    folder = os.path.join(
        DATASET_PATH,
        label
    )

    for file in os.listdir(folder):

        file_path = os.path.join(
            folder,
            file
        )

        try:

            features = extract_features(
                file_path
            )

            X.append(features)

            if label == "real":
                y.append(0)
            else:
                y.append(1)

        except Exception as e:

            print("Error processing:", file)


X = np.array(X)
y = np.array(y)

print("Feature shape:", X.shape)
print("Labels shape:", y.shape)

# Save features
with open(
    "features.pkl",
    "wb"
) as f:

    pickle.dump(
        (X, y),
        f
    )

print("Feature extraction complete.")