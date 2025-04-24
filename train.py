import pandas as pd
import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from tqdm import tqdm

# Paths
meta_path = "data/meta.csv"
keypoints_path = "data/keypoints.h5"

# Parámetros
MAX_SEQ_LEN = 60  # Máximo de frames por clip (ajustable)
N_KEYPOINTS = 33  # Ajustar según cantidad de keypoints por persona
N_FEATURES = 4    # x, y, z, confidence

# 1. Cargar meta.csv
meta = pd.read_csv(meta_path)

# 2. Cargar keypoints.h5
def load_keypoints(clip_id, signer_id="signer_0"):
    with h5py.File(keypoints_path, "r") as f:
        group_names = [clip_id, clip_id + ".mp4"]
        for group in group_names:
            if group in f and signer_id in f[group]:
                kp = f[group][signer_id]["keypoints"][:]
            elif group in f and "keypoints" in f[group]:
                kp = f[group]["keypoints"][:]
            else:
                continue
            cols = int(kp.shape[1]) if len(kp.shape) > 1 else None
            if len(kp.shape) == 2 and cols == 543 * 4:
                kp = kp.reshape((-1, 543, 4))
                kp = kp[:, :33, :]
            elif len(kp.shape) == 2 and cols == 33 * 4:
                kp = kp.reshape((-1, 33, 4))
            elif len(kp.shape) == 3 and kp.shape[1:] == (33, 4):
                pass
            else:
                return None
            return kp
    return None

# Mostrar algunos ids de meta.csv y keypoints.h5 para depuración
# print("Primeros 5 ids en meta.csv:", meta["id"].head().tolist())
# with h5py.File(keypoints_path, "r") as f:
#     h5_ids = list(f.keys())
#     print("Primeros 5 grupos en keypoints.h5:", h5_ids[:5])

X = []
y = []
labels = meta["label"].unique().tolist()
label2idx = {l: i for i, l in enumerate(labels)}

count = 0
not_found = 0
for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Procesando clips"):
    clip_id = row["id"]
    label = row["label"]
    kp = load_keypoints(clip_id)
    if kp is not None:
        seq = np.zeros((MAX_SEQ_LEN, N_KEYPOINTS, N_FEATURES))
        length = min(len(kp), MAX_SEQ_LEN)
        seq[:length] = kp[:length]
        X.append(seq)
        y.append(label2idx[label])
        count += 1
    else:
        not_found += 1
        if not_found <= 5:
            print(f"Advertencia: No se encontraron keypoints para clip_id: {clip_id}")
print(f"Clips cargados: {count}")
print(f"Clips sin keypoints: {not_found}")

X = np.array(X)
y = np.array(y)

# 4. Definir modelo simple (LSTM)
model = keras.Sequential([
    layers.Input(shape=(MAX_SEQ_LEN, N_KEYPOINTS, N_FEATURES)),
    layers.Reshape((MAX_SEQ_LEN, N_KEYPOINTS * N_FEATURES)),
    layers.Masking(mask_value=0.0),
    layers.LSTM(128),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(labels), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 5. Entrenar
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1)

# 6. Guardar modelo y labels
model.save("lsa_sign_model.keras")
with open("labels.txt", "w", encoding="utf-8") as f:
    for l in labels:
        f.write(l + "\n")

print("Entrenamiento finalizado. Modelo guardado como lsa_sign_model.keras")
