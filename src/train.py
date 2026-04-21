import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from model import build_model


# Load features
with open("features.pkl", "rb") as f:

    X, y = pickle.load(f)


print("Loaded data:", X.shape)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)


# Build model
model = build_model(

    input_shape=(100, 40)

)


# Train model
history = model.fit(

    X_train,
    y_train,

    epochs=15,
    batch_size=16,

    validation_data=(

        X_test,
        y_test

    )

)


# Evaluate
loss, accuracy = model.evaluate(

    X_test,
    y_test

)

print("Final Accuracy:", accuracy)


# Save model
model.save("spoof_model.h5")


# Plot accuracy graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")

plt.legend(

    ['Train', 'Validation']

)

plt.savefig("results/accuracy_graph.png")

plt.show()