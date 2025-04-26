import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("lsa_sign_model.keras")
with open("labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

def predict_from_keypoints(keypoints_path):
    keypoints = np.load(keypoints_path)
    if keypoints.shape != (60, 33, 4):
        pad = np.zeros((60, 33, 4))
        length = min(len(keypoints), 60)
        pad[:length] = keypoints[:length]
        keypoints = pad
    keypoints = np.expand_dims(keypoints, axis=0)
    pred = model.predict(keypoints)
    idx = np.argmax(pred)
    return labels[idx]

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python predict.py <keypoints.npy>")
        exit(1)
    keypoints_path = sys.argv[1]
    pred = predict_from_keypoints(keypoints_path)
    print(f"Predicción: {pred}")
