import pandas as pd
import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from tqdm import tqdm
from collections import Counter

# Paths
meta_path = "data/meta.csv"
keypoints_path = "data/keypoints.h5"

# Cargar meta antes de cualquier uso
meta = pd.read_csv(meta_path)

# Agrupar los labels por la primera palabra (o gesto principal)
def extraer_label_simple(frase):
    # Puedes mejorar esta función para extraer la palabra clave relevante
    return frase.strip().split()[0].lower()

meta['label_simple'] = meta['label'].apply(extraer_label_simple)
conteo = Counter(meta['label_simple'])
clases_validas = [c for c, n in conteo.items() if n >= 10]
meta = meta[meta['label_simple'].isin(clases_validas)].reset_index(drop=True)
print(f"Clases simples válidas (>=10 ejemplos): {len(clases_validas)}")

# Parámetros
MAX_SEQ_LEN = 60  # Máximo de frames por clip
N_KEYPOINTS = 33
N_FEATURES = 4    # x, y, z, confidence

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

labels = meta["label_simple"].unique().tolist()
label2idx = {l: i for i, l in enumerate(labels)}

X = []
y = []
count = 0
not_found = 0
for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Procesando clips"):
    clip_id = row["id"]
    label = row["label_simple"]
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

model = keras.Sequential([
    layers.Input(shape=(MAX_SEQ_LEN, N_KEYPOINTS, N_FEATURES)),
    layers.Reshape((MAX_SEQ_LEN, N_KEYPOINTS * N_FEATURES)),
    layers.Masking(mask_value=0.0),
    layers.LSTM(128),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(labels), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X, y, epochs=200, batch_size=64, validation_split=0.1)

model.save("lsa_sign_model.keras")
with open("labels.txt", "w", encoding="utf-8") as f:
    for l in labels:
        f.write(l + "\n")

print("Entrenamiento finalizado. Modelo guardado como lsa_sign_model.keras")

label_counts = Counter(y)
print(f"Cantidad de clases usadas en el entrenamiento: {len(label_counts)}")
print("Ejemplos por clase (top 20):")
for label_idx, count in label_counts.most_common(20):
    print(f"  {labels[label_idx]}: {count}")
# Si quieres ver todas las clases, elimina el [:20]
