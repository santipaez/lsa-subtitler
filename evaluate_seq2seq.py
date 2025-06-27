import numpy as np
import pickle
from tensorflow.keras.models import load_model
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from jiwer import wer
import random

# --- Configuración ---
CACHE_PATH = 'data/seq2seq_train_data.npz'
TOKENIZER_PATH = 'tokenizer.pkl'
MODEL_PATH = 'lsa_seq2seq_model.keras'
MAX_TARGET_LEN = 20
NUM_EXAMPLES = 10  # Ejemplos a mostrar

# --- Cargar datos procesados ---
data = np.load(CACHE_PATH)
X = data['X']
y = data['y']
y_decoder_input = data['y_decoder_input']

# Usar el mismo split que en entrenamiento (último 10% para validación)
val_split = int(0.9 * len(X))
X_val = X[val_split:]
y_val = y[val_split:]

# --- Cargar tokenizer ---
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
index_word = tokenizer.index_word
word_index = tokenizer.word_index

# --- Cargar modelo ---
model = load_model(MODEL_PATH)

# --- Frases ground truth ---
# Si tienes las frases originales, cárgalas aquí, si no, solo muestra los tokens
try:
    import pandas as pd
    meta = pd.read_csv('data/meta.csv')
    frases = meta['label'].astype(str).tolist()
    frases = [f.lower() for f in frases]
    frases = ["<start> " + f + " <end>" for f in frases]
    frases_val = frases[val_split:]
except Exception:
    frases_val = None

# --- Evaluación BLEU/WER y ejemplos ---
print("\nEJEMPLOS DE TRADUCCIÓN (ground truth vs. predicción):")
for i in random.sample(range(len(X_val)), min(NUM_EXAMPLES, len(X_val))):
    enc_input = X_val[i:i+1]
    target_seq = np.zeros((1, MAX_TARGET_LEN-1))
    target_seq[0, 0] = word_index['<start>']
    decoded_sentence = []
    for t in range(MAX_TARGET_LEN-1):
        output_tokens = model.predict([enc_input, target_seq], verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, t, :])
        sampled_word = index_word.get(sampled_token_index, '<unk>')
        if sampled_word == '<end>' or sampled_word == '<unk>':
            break
        decoded_sentence.append(sampled_word)
        if t + 1 < MAX_TARGET_LEN-1:
            target_seq[0, t+1] = sampled_token_index
    if frases_val is not None:
        gt = frases_val[i].replace('<start>','').replace('<end>','').strip()
    else:
        gt = ' '.join([index_word.get(tok, '<unk>') for tok in y_val[i] if tok > 0])
    pred = ' '.join(decoded_sentence)
    bleu = sentence_bleu([gt.split()], pred.split(), smoothing_function=SmoothingFunction().method1)
    wer_score = wer(gt, pred)
    print(f"GT: {gt}\nPRED: {pred}\nBLEU: {bleu:.3f} | WER: {wer_score:.3f}\n---")

# BLEU y WER promedio en validación
gt_list = []
pred_list = []
for i in range(len(X_val)):
    enc_input = X_val[i:i+1]
    target_seq = np.zeros((1, MAX_TARGET_LEN-1))
    target_seq[0, 0] = word_index['<start>']
    decoded_sentence = []
    for t in range(MAX_TARGET_LEN-1):
        output_tokens = model.predict([enc_input, target_seq], verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, t, :])
        sampled_word = index_word.get(sampled_token_index, '<unk>')
        if sampled_word == '<end>' or sampled_word == '<unk>':
            break
        decoded_sentence.append(sampled_word)
        if t + 1 < MAX_TARGET_LEN-1:
            target_seq[0, t+1] = sampled_token_index
    if frases_val is not None:
        gt = frases_val[i].replace('<start>','').replace('<end>','').strip()
    else:
        gt = ' '.join([index_word.get(tok, '<unk>') for tok in y_val[i] if tok > 0])
    pred = ' '.join(decoded_sentence)
    gt_list.append(gt)
    pred_list.append(pred)
    
bleu_scores = [sentence_bleu([gt.split()], pred.split(), smoothing_function=SmoothingFunction().method1) for gt, pred in zip(gt_list, pred_list)]
wer_scores = [wer(gt, pred) for gt, pred in zip(gt_list, pred_list)]
print(f"\nBLEU promedio validación: {np.mean(bleu_scores):.3f}")
print(f"WER promedio validación: {np.mean(wer_scores):.3f}")
