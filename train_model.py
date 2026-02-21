"""
Train brain tumor detection model (binary: tumor vs no tumor).
Supports:
  - dataset/yes and dataset/no (flat)
  - dataset/Training/yes, dataset/Training/no and dataset/Testing/yes, dataset/Testing/no
  - dataset/Training/{glioma,meningioma,pituitary,notumor}, dataset/Testing/...
"""
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
os.makedirs(MODELS_DIR, exist_ok=True)
IMG_SIZE = 128


def _load_folder(base, foldername, label):
    """Load images from a folder; label 1 = tumor, 0 = no tumor."""
    folder = os.path.join(base, foldername)
    if not os.path.exists(folder):
        return [], []
    X, y = [], []
    for f in os.listdir(folder):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            X.append(np.array(img))
            y.append(label)
    return X, y


def load_data():
    """Load dataset. Prefer yes/no layout; fall back to glioma/meningioma/pituitary/notumor."""
    train_path = os.path.join(DATASET_DIR, "Training")
    test_path = os.path.join(DATASET_DIR, "Testing")
    yes_train = os.path.join(train_path, "yes")
    no_train = os.path.join(train_path, "no")
    yes_flat = os.path.join(DATASET_DIR, "yes")
    no_flat = os.path.join(DATASET_DIR, "no")

    # 1) dataset/Training/yes, dataset/Training/no (and optional Testing)
    if os.path.isdir(yes_train) and os.path.isdir(no_train):
        X1, y1 = _load_folder(train_path, "yes", 1)
        X2, y2 = _load_folder(train_path, "no", 0)
        X_train, y_train = X1 + X2, y1 + y2
        yes_test = os.path.join(test_path, "yes")
        no_test = os.path.join(test_path, "no")
        if os.path.isdir(test_path) and os.path.isdir(yes_test) and os.path.isdir(no_test):
            X3, y3 = _load_folder(test_path, "yes", 1)
            X4, y4 = _load_folder(test_path, "no", 0)
            X_test, y_test = X3 + X4, y3 + y4
        else:
            X_test, y_test = X_train, y_train  # no test split
        return _normalize(X_train, y_train, X_test, y_test)

    # 2) dataset/yes and dataset/no (flat; split 80/20 for train/test)
    if os.path.isdir(yes_flat) and os.path.isdir(no_flat):
        X_yes, y_yes = _load_folder(DATASET_DIR, "yes", 1)
        X_no, y_no = _load_folder(DATASET_DIR, "no", 0)
        if not X_yes and not X_no:
            raise FileNotFoundError(f"No images in {yes_flat} or {no_flat}.")
        X_all = np.array(X_yes + X_no, dtype=np.float32) / 255.0
        y_all = np.array(y_yes + y_no, dtype=np.float32)
        n = len(y_all)
        idx = np.random.RandomState(42).permutation(n)
        split = int(0.8 * n)
        train_idx, test_idx = idx[:split], idx[split:]
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]
        return (X_train, y_train), (X_test, y_test)

    # 3) Original: glioma, meningioma, pituitary, notumor
    if not os.path.isdir(train_path) or not os.path.isdir(test_path):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_DIR}. Use either:\n"
            "  - dataset/yes and dataset/no, or\n"
            "  - dataset/Training/yes and dataset/Training/no (and optionally Testing/yes, Testing/no), or\n"
            "  - dataset/Training and dataset/Testing with glioma, meningioma, pituitary, notumor."
        )
    tumor_folders = ["glioma", "meningioma", "pituitary"]
    no_tumor_folder = "notumor"
    X_train, y_train = [], []
    for folder in tumor_folders:
        X, y = _load_folder(train_path, folder, 1)
        X_train.extend(X); y_train.extend(y)
    X, y = _load_folder(train_path, no_tumor_folder, 0)
    X_train.extend(X); y_train.extend(y)
    X_test, y_test = [], []
    for folder in tumor_folders:
        X, y = _load_folder(test_path, folder, 1)
        X_test.extend(X); y_test.extend(y)
    X, y = _load_folder(test_path, no_tumor_folder, 0)
    X_test.extend(X); y_test.extend(y)
    return _normalize(X_train, y_train, X_test, y_test)


def _normalize(X_train, y_train, X_test, y_test):
    X_train = np.array(X_train, dtype=np.float32) / 255.0
    y_train = np.array(y_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32) / 255.0
    y_test = np.array(y_test, dtype=np.float32)
    return (X_train, y_train), (X_test, y_test)


def build_model():
    base = tf.keras.applications.VGG16(
        input_shape=(128, 128, 3),
        include_top=False,
        weights="imagenet",
    )
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers[-4:]:
        layer.trainable = True

    x = base.output
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _class_weights(y_train):
    """Compute class weights so both yes (1) and no (0) are learned; prevents always predicting one class."""
    n_total = len(y_train)
    n_tumor = int(np.sum(y_train))
    n_no_tumor = n_total - n_tumor
    if n_tumor == 0 or n_no_tumor == 0:
        return None
    # Weight inversely proportional to class frequency
    w_tumor = n_total / (2.0 * n_tumor)
    w_no_tumor = n_total / (2.0 * n_no_tumor)
    return {0: w_no_tumor, 1: w_tumor}


def main():
    print("Loading dataset...")
    (X_train, y_train), (X_test, y_test) = load_data()
    n_train, n_test = len(X_train), len(X_test)
    n_yes = int(np.sum(y_train))
    n_no = n_train - n_yes
    print(f"Train: {n_train} (yes/tumor: {n_yes}, no: {n_no}), Test: {n_test}")
    if n_yes == 0 or n_no == 0:
        raise ValueError("Need both yes (tumor) and no (no tumor) images in the dataset.")
    class_w = _class_weights(y_train)
    if class_w:
        print(f"Class weights: no_tumor={class_w[0]:.3f}, tumor={class_w[1]:.3f} (balanced training)")
    print("Building model...")
    model = build_model()
    print("Training... (yes=tumor, no=no tumor)")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=16,
        class_weight=class_w,
        shuffle=True,
        callbacks=callbacks,
    )
    model.save(os.path.join(MODELS_DIR, "brain_tumor_model.keras"))
    with open(os.path.join(MODELS_DIR, "label_convention.txt"), "w") as f:
        f.write("yes_is_tumor=1\n")
    print("Model saved to", MODELS_DIR, "(yes=tumor, no=no tumor)")


if __name__ == "__main__":
    main()
