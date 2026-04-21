import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model


# Load features
with open("features.pkl", "rb") as f:

    X, y = pickle.load(f)


# Split again
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)


# Load trained model
model = load_model("spoof_model.h5")


# Predict
y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5).astype(int)


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# Plot matrix
plt.figure(figsize=(6,5))

sns.heatmap(

    cm,
    annot=True,
    fmt="d",
    cmap="Blues"

)

plt.title("Confusion Matrix")

plt.ylabel("Actual")

plt.xlabel("Predicted")

plt.savefig("results/confusion_matrix.png")

plt.show()