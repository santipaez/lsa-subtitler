try:
    import pandas as pd
except ImportError:
    # Fall back to a simple CSV reader if pandas is not available
    import csv
    def read_csv(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data

import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import os
from collections import Counter, deque
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from jiwer import wer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Paths
meta_path = "data/meta.csv"
keypoints_path = "data/keypoints.h5"

# Hiperparámetros
MAX_SEQ_LEN = 30  # Máximo de frames por clip
N_KEYPOINTS = 33
N_FEATURES = 3    # x, y, confidence
NUM_WORDS = 25000  # Tamaño del vocabulario del texto
MAX_TARGET_LEN = 30  # Máximo de palabras por frase

# Arquitectura del modelo simplificada para mejor generalización
EMBEDDING_DIM = 128     # Reducido para mejor rendimiento y evitar overfitting
ENCODER_UNITS = 128     # Reducido para evitar overfitting
DECODER_UNITS = 128     # Reducido para evitar overfitting
ATTENTION_UNITS = 128   # Reducido para evitar overfitting
ENCODER_LAYERS = 1      # Simplificado a una sola capa
DECODER_LAYERS = 1      # Simplificado a una sola capa
USE_ATTENTION = True    # Mantener mecanismo de atención
USE_BIDIRECTIONAL = True # Mantener bidireccionalidad para mejor contexto

# Hiperparámetros de entrenamiento optimizados
BATCH_SIZE = 16        # Reducido para menor uso de memoria y mejor generalización
EPOCHS = 300            # Reducido para evitar overfitting
INITIAL_LR = 1e-3       # Learning rate inicial ajustado
MIN_LR = 1e-5           # Learning rate mínimo
CLIP_NORM = 1.0         # Clipping de gradientes para estabilidad
PATIENCE = 10            # Paciencia para early stopping reducida

# Hiperparámetros de regularización ajustados
DROPOUT_RATE = 0.3      # Aumentado para mejorar generalización
RECURRENT_DROPOUT = 0.2 # Aumentado para regularizar LSTMs
EMBEDDING_DROPOUT = 0.2 # Aumentado para embeddings
L2_REG = 1e-4           # Aumentado para mejor regularización
FOCAL_LOSS_GAMMA = 2.0  # Factor para focal loss

# Hiperparámetros de teacher forcing ajustados
TEACHER_FORCING_RATIO_INITIAL = 1.0  # Comenzar con 100% teacher forcing
TEACHER_FORCING_RATIO_FINAL = 0.5    # Terminar con 50% teacher forcing

# Parámetros para data augmentation
MAX_AUGMENTATIONS = 2   # Número máximo de augmentaciones por ejemplo
APPLY_AUGMENTATION = True  # Activar/desactivar augmentación

# Parámetros para muestreo en inferencia
SAMPLING_TEMPERATURE = 0.3  # Temperatura reducida para menor aleatoriedad
SAMPLING_TOPK = 5          # Top-k reducido para mayor precisión
USE_NUCLEUS_SAMPLING = True # Usar nucleus sampling
NUCLEUS_TOP_P = 0.8        # Valor restrictivo para nucleus sampling

# Si quieres debug rápido, ajusta este valor (>0 para usar subset)
DEBUG_SUBSET_SIZE = 0  # Cambia a 0 para usar todo el dataset

# Función mejorada de normalización espacial de keypoints
def normalize_keypoints(keypoints):
    """
    Normaliza espacialmente los keypoints para mejorar la invariancia a la posición y escala.
    
    Args:
        keypoints: Array de keypoints de forma (frames, n_keypoints, n_features)
    
    Returns:
        Array normalizado de keypoints
    """
    # Crear copia para no modificar el original
    norm_keypoints = keypoints.copy()
    
    # Para cada frame, normalizar espacialmente
    for i in range(norm_keypoints.shape[0]):
        # Encontrar los keypoints con confianza suficiente en este frame
        confidence = norm_keypoints[i, :, 2]
        valid_mask = confidence > 0.3
        
        if np.sum(valid_mask) < 5:  # Si hay muy pocos keypoints válidos, omitir normalización
            continue
            
        valid_keypoints = norm_keypoints[i, valid_mask, :2]
        
        # 1. Centrar usando el punto medio entre los hombros (keypoints 11 y 12)
        # Si los hombros no están disponibles, usar el centroide de todos los puntos válidos
        if valid_mask[11] and valid_mask[12]:  # Si ambos hombros son válidos
            center = (norm_keypoints[i, 11, :2] + norm_keypoints[i, 12, :2]) / 2
        else:
            center = np.mean(valid_keypoints, axis=0)
            
        # Centrar todos los keypoints
        norm_keypoints[i, :, 0] = norm_keypoints[i, :, 0] - center[0]
        norm_keypoints[i, :, 1] = norm_keypoints[i, :, 1] - center[1]
        
        # 2. Escalar basado en la distancia media desde el centro
        # Esto hace que la escala sea invariante al tamaño del cuerpo
        distances = np.sqrt(np.sum(valid_keypoints**2, axis=1))
        scale_factor = np.mean(distances) + 1e-6  # Evitar división por cero
        
        # Aplicar escalado
        norm_keypoints[i, :, 0] = norm_keypoints[i, :, 0] / scale_factor
        norm_keypoints[i, :, 1] = norm_keypoints[i, :, 1] / scale_factor
    
    return norm_keypoints

# Función para aumentar los datos con transformaciones
def augment_keypoints(keypoints, max_augmentations=2):
    """
    Aplica técnicas de data augmentation a los keypoints.
    
    Args:
        keypoints: Array de keypoints de forma (frames, n_keypoints, n_features)
        max_augmentations: Número máximo de augmentaciones a generar
    
    Returns:
        Lista de arrays de keypoints aumentados
    """
    augmented = []
    
    # Añadir keypoints originales
    augmented.append(keypoints)
    
    # Solo continuar si necesitamos más augmentaciones
    if max_augmentations <= 1:
        return augmented
    
    # 1. Añadir pequeño jitter (ruido gaussiano)
    keypoints_jitter = keypoints.copy()
    # Añadir ruido solo a las coordenadas x,y, no a la confianza
    noise_scale = 0.02
    keypoints_jitter[:, :, 0] += np.random.normal(0, noise_scale, keypoints_jitter[:, :, 0].shape)
    keypoints_jitter[:, :, 1] += np.random.normal(0, noise_scale, keypoints_jitter[:, :, 1].shape)
    augmented.append(keypoints_jitter)
    
    # Si ya tenemos suficientes augmentaciones, devolver
    if len(augmented) >= max_augmentations:
        return augmented
    
    # 2. Espejado horizontal (invertir eje x)
    # Esto es útil para señas no direccionales
    keypoints_flipped = keypoints.copy()
    keypoints_flipped[:, :, 0] = -keypoints_flipped[:, :, 0]  # Invertir coordenada x
    
    # Intercambiar pares izquierda-derecha (hombros, codos, muñecas, etc.)
    # Mapeado de pares de keypoints (izquierda-derecha)
    pairs = [
        (11, 12),  # Hombros
        (13, 14),  # Codos
        (15, 16),  # Muñecas
        (23, 24),  # Caderas
        (25, 26),  # Rodillas
        (27, 28),  # Tobillos
    ]
    
    for left, right in pairs:
        if left < keypoints.shape[1] and right < keypoints.shape[1]:
            # Intercambiar keypoints izquierda-derecha
            keypoints_flipped[:, [left, right]] = keypoints_flipped[:, [right, left]]
    
    augmented.append(keypoints_flipped)
    
    # Limitar al número máximo de augmentaciones
    return augmented[:max_augmentations]

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
                kp = kp[:, :, [0, 1, 3]]
            elif len(kp.shape) == 2 and cols == 33 * 4:
                kp = kp.reshape((-1, 33, 4))
                kp = kp[:, :, [0, 1, 3]]
            elif len(kp.shape) == 3 and kp.shape[1:] == (33, 4):
                kp = kp[:, :, [0, 1, 3]]
            else:
                return None
            return kp
    return None

# Cargar meta y usar TODO el dataset para traducción libre
meta = pd.read_csv(meta_path)
print(f"Total de clips en el dataset: {len(meta)}")
# Ahora no se filtran frases por frecuencia: el modelo aprende a traducir cualquier frase (traducción libre, no clasificación de frases fijas)

# Tokenización de frases
frases = meta['label'].astype(str).tolist()
frases = [f.lower() for f in frases]
frases = ["<start> " + f + " <end>" for f in frases]

# DIAGNÓSTICO: Ver el texto original
print("\n--- DIAGNÓSTICO TOKENIZER ---")
print(f"Ejemplos de frases originales:\n{frases[:5]}")

# Crear el tokenizer con un tamaño de vocabulario suficiente
tokenizer = Tokenizer(num_words=NUM_WORDS, filters='', oov_token='<unk>')
tokenizer.fit_on_texts(frases)
word_index = tokenizer.word_index
vocab_size = min(NUM_WORDS, len(word_index) + 1)

# DIAGNÓSTICO: Comprobar el tamaño real del vocabulario
print(f"Total palabras únicas encontradas: {len(word_index)}")
print(f"Tamaño de vocabulario usado (num_words): {vocab_size}")
print(f"Índice de <start>: {word_index.get('<start>')}")
print(f"Índice de <end>: {word_index.get('<end>')}")
print(f"Índice de <unk>: {word_index.get('<unk>')}")

# Convertir el texto a secuencias
sequences = tokenizer.texts_to_sequences(frases)

# DIAGNÓSTICO: Ver ejemplos de secuencias tokenizadas
print(f"\nEjemplos de secuencias:\n{sequences[:5]}")

# Contar cuántas secuencias tienen <unk>
unk_index = word_index.get('<unk>', 1)
unk_counts = [seq.count(unk_index) for seq in sequences[:100]]
print(f"Cantidad de <unk> en primeras 100 secuencias: {sum(unk_counts)}")
print(f"Promedio de <unk> por secuencia: {sum(unk_counts)/100:.2f}")

# Si hay muchos <unk>, puede ser problema del tamaño del vocabulario
if sum(unk_counts)/100 > 1.0:
    print("ADVERTENCIA: Muchos tokens <unk> - considera aumentar NUM_WORDS")

# --- PROCESAMIENTO DE DATOS ---
import os
from tqdm import tqdm
import h5py

print('Procesando datos (esto puede tardar)...')
X = []
y = []
y_decoder_input = []
clips_sin_keypoints = 0
primeros_kp = []

# Implementar curriculum learning para el entrenamiento
def apply_curriculum_learning(X, y, y_decoder_input, epoch):
    """
    Aplica curriculum learning: comenzar con ejemplos más simples y aumentar gradualmente
    la complejidad durante el entrenamiento.
    
    Args:
        X: Datos de entrada (keypoints)
        y: Datos objetivo (tokens de salida)
        y_decoder_input: Datos de entrada para el decoder
        epoch: Época actual de entrenamiento
    
    Returns:
        Subconjunto de datos para esta etapa de entrenamiento
    """
    if epoch < 5:
        # En las primeras épocas, usar secuencias más cortas
        max_seq_len = min(5 + epoch * 3, MAX_TARGET_LEN)
        
        # Contar tokens distintos de cero en cada secuencia objetivo
        seq_lengths = np.array([np.sum(seq > 0) for seq in y])
        
        # Filtrar ejemplos según su longitud
        mask = seq_lengths <= max_seq_len
        
        # Aplicar máscara para seleccionar ejemplos
        X_filtered = X[mask]
        y_filtered = y[mask]
        y_decoder_input_filtered = y_decoder_input[mask]
        
        print(f"Curriculum learning (época {epoch}): usando {len(X_filtered)} ejemplos con longitud <= {max_seq_len}")
        return X_filtered, y_filtered, y_decoder_input_filtered
    
    # Después de las primeras épocas, usar todos los datos
    return X, y, y_decoder_input

# Si estamos en modo debug, limitar el número de ejemplos
proc_meta = meta
if DEBUG_SUBSET_SIZE > 0:
    proc_meta = meta.head(DEBUG_SUBSET_SIZE * 2)  # Multiplicamos por 2 para compensar posibles clips sin keypoints

for idx, row in tqdm(proc_meta.iterrows(), total=len(proc_meta), desc='Procesando clips', 
                    bar_format='{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                    mininterval=1.0, maxinterval=5.0, ncols=80, leave=False):
    clip_id = row['id']
    signer_id = row.get('signer_id', 'signer_0')
    kp = load_keypoints(clip_id, signer_id)
    
    # Verificar si los keypoints son válidos
    if kp is not None and not np.isnan(kp).all():
        # Aplicar normalización espacial a los keypoints
        kp = normalize_keypoints(kp)
        
        # Aplicar data augmentation si está activado
        if APPLY_AUGMENTATION:
            augmented_kps = augment_keypoints(kp, max_augmentations=MAX_AUGMENTATIONS)
        else:
            augmented_kps = [kp]  # Solo usar el original
        
        # Procesar cada versión aumentada de los keypoints
        for kp_aug in augmented_kps:
            # Padding o truncado de frames
            seq = np.zeros((MAX_SEQ_LEN, N_KEYPOINTS, N_FEATURES), dtype=np.float32)
            length = min(len(kp_aug), MAX_SEQ_LEN)
            seq[:length] = kp_aug[:length]
            X.append(seq)
            
            # Guardar algunos ejemplos para visualización
            if len(primeros_kp) < 3:
                primeros_kp.append(seq)
                
            # Target para decoder (ya tokenizado y padded)
            seq_tokens = sequences[idx]
            
            # Preparar secuencias de entrada y salida del decoder
            tgt = seq_tokens[1:]  # sin <start>
            tgt_in = seq_tokens[:-1]  # sin <end>
            
            # Aplicar padding a secuencias objetivo
            if len(tgt) < MAX_TARGET_LEN - 1:
                tgt = tgt + [0] * ((MAX_TARGET_LEN - 1) - len(tgt))
            else:
                tgt = tgt[:MAX_TARGET_LEN - 1]
                
            # Aplicar padding a secuencias de entrada
            if len(tgt_in) < MAX_TARGET_LEN - 1:
                tgt_in = tgt_in + [0] * ((MAX_TARGET_LEN - 1) - len(tgt_in))
            else:
                tgt_in = tgt_in[:MAX_TARGET_LEN - 1]
                
            # Almacenar secuencias procesadas
            y.append(tgt)
            y_decoder_input.append(tgt_in)
        
        # Si estamos en modo debug y alcanzamos el límite, salir
        if DEBUG_SUBSET_SIZE > 0 and len(X) >= DEBUG_SUBSET_SIZE:
            break
    else:
        clips_sin_keypoints += 1
        if clips_sin_keypoints <= 5:
            print(f"Advertencia: No se encontraron keypoints para clip_id: {clip_id}")

# Convertir listas a arrays de NumPy
X = np.array(X)
y = np.array(y)
y_decoder_input = np.array(y_decoder_input)

# Mostrar información del dataset y efectos de la aumentación
print(f'Clips cargados: {len(X)} (incluidas {len(X) - len(proc_meta) + clips_sin_keypoints} aumentaciones)')
print(f'Clips sin keypoints: {clips_sin_keypoints}')

if len(X) == 0:
    raise RuntimeError('No se cargó ningún clip válido con keypoints. Revisa el archivo keypoints.h5 y la función load_keypoints.')

print('Shape de X:', X.shape)
print('Shape de y_decoder_input:', y_decoder_input.shape)
print('Shape de y_decoder_target:', y.shape)

# Mostrar algunos ejemplos de keypoints
for i, kp in enumerate(primeros_kp):
    print(f'Keypoints ejemplo {i}:', kp[:2])

# Verificar si necesitamos aplicar el modo debug
if DEBUG_SUBSET_SIZE > 0 and len(X) > DEBUG_SUBSET_SIZE:
    # Seleccionar un subconjunto aleatorio de datos
    indices = np.random.choice(len(X), DEBUG_SUBSET_SIZE, replace=False)
    X = X[indices]
    y = y[indices]
    y_decoder_input = y_decoder_input[indices]
    print(f"Usando solo {DEBUG_SUBSET_SIZE} ejemplos para debug rápido.")

# Obtener algunas frases originales para mostrar durante evaluación
frases_ejemplos = frases[:min(20, len(frases))]

print("Implementando scheduled sampling para reducir teacher forcing gradualmente")

# División train/val (80% / 20%)
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]
y_decoder_input_train, y_decoder_input_val = y_decoder_input[:train_size], y_decoder_input[train_size:]

# Guardar algunas frases originales para validación
# Obtenemos los índices de validación real basados en la división de train/val
val_indices = list(range(len(meta)))[-len(X_val):]
# Extraemos las frases correspondientes a esos índices
frases_val = [frases[i] for i in val_indices if i < len(frases)]

# Asegurar shapes consistentes
print(f"Shape de X: {X.shape}")
print(f"Shape de y_decoder_input: {y_decoder_input.shape}")
print(f"Shape de y_decoder_target: {y.shape}")

# Verificar que las dimensiones sean correctas y aplicar padding si es necesario
if len(y_decoder_input.shape) < 2:
    print("Error: y_decoder_input no tiene la dimensión correcta")
    y_decoder_input = np.array(y_decoder_input).reshape(-1, 1)

if len(y.shape) < 2:
    print("Error: y no tiene la dimensión correcta")
    y = np.array(y).reshape(-1, 1)

# Verificar longitud de secuencias y aplicar padding si es necesario
if y_decoder_input.shape[1] != MAX_TARGET_LEN - 1:
    print(f"Ajustando dimensión de y_decoder_input a {MAX_TARGET_LEN - 1}")
    padded_input = np.zeros((y_decoder_input.shape[0], MAX_TARGET_LEN - 1), dtype=int)
    # Copiar los datos existentes, limitando a MAX_TARGET_LEN - 1
    for i in range(len(y_decoder_input)):
        seq_len = min(len(y_decoder_input[i]), MAX_TARGET_LEN - 1)
        padded_input[i, :seq_len] = y_decoder_input[i][:seq_len]
    y_decoder_input = padded_input
    y_decoder_input_train = y_decoder_input[:train_size]
    y_decoder_input_val = y_decoder_input[train_size:]

if y.shape[1] != MAX_TARGET_LEN - 1:
    print(f"Ajustando dimensión de y a {MAX_TARGET_LEN - 1}")
    padded_target = np.zeros((y.shape[0], MAX_TARGET_LEN - 1), dtype=int)
    # Copiar los datos existentes, limitando a MAX_TARGET_LEN - 1
    for i in range(len(y)):
        seq_len = min(len(y[i]), MAX_TARGET_LEN - 1)
        padded_target[i, :seq_len] = y[i][:seq_len]
    y = padded_target
    y_train = y[:train_size]
    y_val = y[train_size:]
    
print(f"Shape final de X_train: {X_train.shape}, X_val: {X_val.shape}")
print(f"Shape final de y_decoder_input_train: {y_decoder_input_train.shape}, y_decoder_input_val: {y_decoder_input_val.shape}")
print(f"Shape final de y_train: {y_train.shape}, y_val: {y_val.shape}")

# Chequeo adicional de datos
print("Chequeando datos de entrada...")
print(f"NaN en X: {np.isnan(X).sum()} | inf en X: {np.isinf(X).sum()} | max: {np.nanmax(X)}, min: {np.nanmin(X)}")
print(f"NaN en y: {np.isnan(y).sum()} | inf: {np.isinf(y).sum()} | max: {np.max(y)}, min: {np.min(y)}")

# Analizar y limpiar los datos
# Reemplazar NaN/inf por ceros
if np.isnan(X).sum() > 0 or np.isinf(X).sum() > 0:
    print(f"Se encontraron valores problemáticos en X: {np.isnan(X).sum()} NaN, {np.isinf(X).sum()} Inf")
    print("Limpiando datos...")
    
    # Localizar keypoints problemáticos
    nan_rows = np.isnan(X).any(axis=(2, 3)).sum(axis=1)
    print(f"Número de frames con NaN por secuencia (top 10): {sorted(nan_rows)[-10:]}")
    
    # Estrategia de interpolación para valores faltantes antes de reemplazar con ceros
    for i in range(X.shape[0]):  # Para cada clip
        for k in range(N_KEYPOINTS):  # Para cada keypoint
            for f in range(N_FEATURES):  # Para cada característica
                # Extraer la serie temporal para este keypoint/característica
                series = X[i, :, k, f]
                nan_mask = np.isnan(series)
                if nan_mask.any() and not nan_mask.all():  # Si hay algunos NaN pero no todos
                    # Interpolar valores faltantes
                    valid_indices = np.where(~nan_mask)[0]
                    valid_values = series[valid_indices]
                    all_indices = np.arange(len(series))
                    # Interpolación lineal
                    interp_values = np.interp(all_indices, valid_indices, valid_values)
                    X[i, :, k, f] = interp_values
    
    # Después de intentar interpolación, reemplazar cualquier NaN/inf restante con ceros
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print("Se completó la limpieza de datos.")
    print(f"Post-limpieza: NaN en X: {np.isnan(X).sum()} | inf en X: {np.isinf(X).sum()} | max: {np.nanmax(X)}, min: {np.nanmin(X)}")

# Recortar valores extremos para evitar explosiones numéricas
X_std = np.std(X[~np.isnan(X)])
X_mean = np.mean(X[~np.isnan(X)])
X_threshold = X_mean + 5 * X_std  # 5 desviaciones estándar
X = np.clip(X, -X_threshold, X_threshold)
print(f"Datos recortados a rango [{-X_threshold:.2f}, {X_threshold:.2f}] para estabilidad numérica")

# Imprimir ejemplos para debugging
print(f"Ejemplo X[0]: {X[0]}")
print(f"Ejemplo X[1]: {X[1]}")

# Definir clase para mecanismo de atención Bahdanau
@keras.utils.register_keras_serializable()
class BahdanauAttention(keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = keras.layers.Dense(units, use_bias=False)
        self.W2 = keras.layers.Dense(units, use_bias=False)
        self.V = keras.layers.Dense(1, use_bias=False)
        
    def call(self, query, values):
        # query shape: (batch_size, hidden_size)
        # values shape: (batch_size, seq_len, hidden_size)
        
        # Expandir query para cálculo de atención
        # query_expanded shape: (batch_size, 1, hidden_size)
        query_expanded = tf.expand_dims(query, 1)
        
        # Calcular score de atención
        # score shape: (batch_size, seq_len, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query_expanded)))
        
        # Obtener pesos de atención
        # attention_weights shape: (batch_size, seq_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Aplicar pesos al encoder output
        # context_vector shape: (batch_size, hidden_size)
        context_vector = tf.reduce_sum(attention_weights * values, axis=1)
        
        return context_vector, attention_weights

# Input layers
encoder_inputs = keras.layers.Input(shape=(MAX_SEQ_LEN, N_KEYPOINTS, N_FEATURES), name="encoder_inputs")

# Crear capa de normalización para la entrada
if len(X.shape) == 4 and X.shape[3] == N_FEATURES:
    print("Aplicando normalización a datos de entrada...")
    
    # Primero normalizar los datos de entrada (pre-procesamiento)
    print("Pre-procesamiento: Normalizando datos de entrada...")
    # Calcular estadísticas solo en valores no-NaN
    X_flat = X.reshape(-1, N_FEATURES)
    mask = ~np.isnan(X_flat).any(axis=1)
    X_valid = X_flat[mask]
    
    if len(X_valid) > 0:
        # Calcular media y desviación estándar por característica
        mean_vals = np.mean(X_valid, axis=0)
        std_vals = np.std(X_valid, axis=0)
        std_vals[std_vals < 1e-10] = 1.0  # Evitar división por cero
        
        # Aplicar normalización z-score
        X_normalized = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    if not np.isnan(X[i, j, k]).any():
                        X_normalized[i, j, k] = (X[i, j, k] - mean_vals) / std_vals
        
        # Reemplazar NaN con ceros después de normalizar
        X = np.nan_to_num(X_normalized, nan=0.0)
        print(f"Normalización aplicada. Media: {mean_vals}, Std: {std_vals}")
    
    # Reshape para normalización adicional en el modelo
    reshape_layer = keras.layers.Reshape((MAX_SEQ_LEN, N_KEYPOINTS * N_FEATURES))
    encoder_outputs = reshape_layer(encoder_inputs)
    
    # Normalización por lotes adicional para estabilidad numérica
    batch_norm = keras.layers.BatchNormalization(name="input_normalization")
    encoder_outputs = batch_norm(encoder_outputs)
    
    # Aplicar masking para manejar secuencias de longitud variable
    mask_layer = keras.layers.Masking(mask_value=0.0, name="masking")(encoder_outputs)
    
    # Agregar capa de masking para ignorar timesteps con puros ceros
    mask_layer = keras.layers.Masking(mask_value=0.0)(encoder_outputs)

# Construir el modelo secuencial con arquitectura simplificada
# Aplicamos un encoder bidireccional single-layer LSTM
bilstm = keras.layers.Bidirectional(
    keras.layers.LSTM(
        ENCODER_UNITS,
        return_sequences=USE_ATTENTION,
        return_state=True,
        dropout=DROPOUT_RATE,
        recurrent_dropout=RECURRENT_DROPOUT,
        name="encoder_lstm"
    )
)(mask_layer)

# Extracting the encoder outputs and states based on bidirectional architecture
if USE_BIDIRECTIONAL:
    # If bidirectional, we get outputs, forward_h, forward_c, backward_h, backward_c
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = bilstm
    # Concatenate states for decoder initialization
    encoder_state_h = keras.layers.Concatenate()([forward_h, backward_h])
    encoder_state_c = keras.layers.Concatenate()([forward_c, backward_c])
    # Project to decoder dimension if needed
    if ENCODER_UNITS * 2 != DECODER_UNITS:
        encoder_state_h = keras.layers.Dense(DECODER_UNITS)(encoder_state_h)
        encoder_state_c = keras.layers.Dense(DECODER_UNITS)(encoder_state_c)
    encoder_states = [encoder_state_h, encoder_state_c]
else:
    # If not bidirectional, we get outputs, state_h, state_c
    encoder_outputs, encoder_state_h, encoder_state_c = bilstm
    encoder_states = [encoder_state_h, encoder_state_c]

# Decoder inputs
decoder_inputs = keras.layers.Input(shape=(MAX_TARGET_LEN - 1,), name='decoder_inputs')
dec_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True, name='embedding_layer')(decoder_inputs)
dec_emb = keras.layers.Dropout(EMBEDDING_DROPOUT)(dec_emb)

# Create attention mechanism if enabled
if USE_ATTENTION:
    attention = BahdanauAttention(ATTENTION_UNITS)
    context_vector, attention_weights = attention(encoder_states[0], encoder_outputs)
    
    # Reshape and expand context vector for merging with decoder input at each timestep
    seq_len = keras.backend.int_shape(dec_emb)[1]
    context_expanded = keras.layers.Lambda(
        lambda x: tf.tile(tf.expand_dims(x, 1), [1, seq_len, 1]),
        output_shape=(seq_len, ENCODER_UNITS * (2 if USE_BIDIRECTIONAL else 1))
    )(context_vector)
    
    # Combine embedded decoder input with context vector
    decoder_combined_input = keras.layers.Concatenate()([dec_emb, context_expanded])
else:
    decoder_combined_input = dec_emb

# Decoder LSTM
decoder_lstm = keras.layers.LSTM(
    DECODER_UNITS,
    return_sequences=True,
    dropout=DROPOUT_RATE,
    recurrent_dropout=RECURRENT_DROPOUT, 
    name="decoder_lstm"
)(decoder_combined_input, initial_state=encoder_states)

# Output layer
decoder_outputs = keras.layers.Dense(vocab_size, activation='softmax', name="output_layer")(decoder_lstm)

# Create model
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Configure optimizer with gradient clipping
optimizer = keras.optimizers.Adam(
    learning_rate=INITIAL_LR,
    clipnorm=CLIP_NORM,  # Limit gradient norm for stability
    epsilon=1e-8  # Increase epsilon for numerical stability
)

# Define loss function and compile model
def focal_loss(y_true, y_pred, gamma=FOCAL_LOSS_GAMMA):
    """
    Implementa Focal Loss para dar más peso a ejemplos difíciles durante el entrenamiento.
    
    Args:
        y_true: Etiquetas verdaderas (sparse - índices de clase)
        y_pred: Probabilidades predichas
        gamma: Factor de enfoque (> 0). Mayor valor = más enfoque en ejemplos difíciles
    
    Returns:
        Loss calculada
    """
    # Handle NaN values
    has_nan = tf.reduce_any(tf.math.is_nan(y_pred))
    tf.cond(has_nan, 
           lambda: tf.print("NaN detected in predictions during loss calculation!"), 
           lambda: tf.no_op())
    
    # Replace NaNs with small values
    y_pred = tf.where(tf.math.is_nan(y_pred), tf.ones_like(y_pred) * 1e-7, y_pred)
    
    # Convert labels to proper format
    y_true_sparse = tf.cast(y_true, tf.int32)
    
    # Create mask to ignore padding tokens (0)
    mask = tf.cast(tf.not_equal(y_true_sparse, 0), tf.float32)
    
    # Get one-hot representation to calculate probabilities
    y_true_one_hot = tf.one_hot(y_true_sparse, depth=tf.shape(y_pred)[-1])
    
    # Calculate probabilities of the true classes
    pt = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
    pt = tf.clip_by_value(pt, 1e-7, 1.0)  # Evitar log(0)
    
    # Apply focal weight: (1-pt)^gamma
    focal_weight = tf.pow(1. - pt, gamma)
    
    # Calculae Cross-Entropy loss
    ce_loss = -tf.math.log(pt)
    
    # Apply focal weights and mask
    loss = focal_weight * ce_loss * mask
    
    # Return mean loss
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-7)

def weighted_loss(y_true, y_pred):
    # Handle NaN values - use tf.where directly
    # Check for NaNs and log for debugging
    has_nan = tf.reduce_any(tf.math.is_nan(y_pred))
    tf.cond(has_nan, 
           lambda: tf.print("NaN detected in predictions during loss calculation!"), 
           lambda: tf.no_op())
    
    # Always apply tf.where to replace NaNs with small values
    y_pred = tf.where(tf.math.is_nan(y_pred), tf.ones_like(y_pred) * 1e-7, y_pred)
    
    # Convert to proper shape for sparse categorical crossentropy
    # If y_true is already in the right format (just integers), we leave it as is
    # Otherwise, we cast it to int32 but don't try to squeeze dimensions
    y_true_sparse = tf.cast(y_true, tf.int32)
    
    # Create mask to ignore padding tokens (0)
    mask = tf.cast(tf.not_equal(y_true_sparse, 0), tf.float32)
    
    # Apply standard categorical crossentropy with masking
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_sparse, y_pred) * mask
    
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + epsilon)

model.compile(
    optimizer=optimizer,
    loss=focal_loss,  # Cambiado de weighted_loss a focal_loss
    metrics=['accuracy']
)

print(model.summary())

# Callback for NaN detection during training
class NanInfCallback(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None and (np.isnan(loss) or np.isinf(loss)):
            print(f'\nNaN/Inf detected in batch {batch}, loss={loss}\n')
            print('Stopping training to prevent further issues...')
            self.model.stop_training = True

# Prepare data for direct feeding (simpler approach to avoid generator issues)
print("\nIniciando entrenamiento con modelo optimizado...")

# Clean any remaining NaN values
X_train = np.nan_to_num(X_train, nan=0.0)
X_val = np.nan_to_num(X_val, nan=0.0)

# Guardar datos originales en el modelo para curriculum learning
model.X_train = X_train
model.y_train = y_train
model.y_decoder_input_train = y_decoder_input_train

# Callback personalizado para curriculum learning
class CurriculumLearningCallback(keras.callbacks.Callback):
    def __init__(self):
        super(CurriculumLearningCallback, self).__init__()
        
    def on_epoch_begin(self, epoch, logs=None):
        # Aplicar curriculum learning para esta época
        X_curr, y_curr, decoder_inp_curr = apply_curriculum_learning(
            self.model.X_train, self.model.y_train, self.model.y_decoder_input_train, epoch
        )
        
        # Guardar las referencias a los datos filtrados para esta época
        self.model.X_train_curr = X_curr
        self.model.y_train_curr = y_curr
        self.model.y_decoder_input_train_curr = decoder_inp_curr
        
        print(f"Curriculum learning: usando {len(X_curr)} ejemplos para época {epoch+1}")

# Modificar el callback de ScheduledSampling para utilizar el curriculum learning
class CombinedCallback(keras.callbacks.Callback):
    def __init__(self, initial_tf_ratio=1.0, final_tf_ratio=0.5):
        super(CombinedCallback, self).__init__()
        self.initial_tf_ratio = initial_tf_ratio
        self.final_tf_ratio = final_tf_ratio
        
    def on_epoch_begin(self, epoch, logs=None):
        # 1. Aplicar curriculum learning
        X_curr, y_curr, decoder_inp_curr = apply_curriculum_learning(
            self.model.X_train, self.model.y_train, self.model.y_decoder_input_train, epoch
        )
        
        # Guardar las referencias a los datos filtrados
        self.model.X_train_curr = X_curr
        self.model.y_train_curr = y_curr
        self.model.y_decoder_input_train_curr = decoder_inp_curr
        
        # 2. Actualizar teacher forcing ratio
        global global_teacher_forcing_ratio
        progress = epoch / (EPOCHS - 1) if EPOCHS > 1 else 1.0
        global_teacher_forcing_ratio = self.initial_tf_ratio - progress * (self.initial_tf_ratio - self.final_tf_ratio)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} - Teacher forcing ratio: {global_teacher_forcing_ratio:.2f}")
        print(f"Curriculum learning: usando {len(X_curr)}/{len(self.model.X_train)} ejemplos")

# Define callbacks
combined_callback = CombinedCallback(
    initial_tf_ratio=TEACHER_FORCING_RATIO_INITIAL,
    final_tf_ratio=TEACHER_FORCING_RATIO_FINAL
)

nan_inf_detection = NanInfCallback()

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=PATIENCE // 2,
    min_lr=MIN_LR,
    verbose=1,
    cooldown=2
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='lsa_seq2seq_best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Use direct batch training instead of generator for simpler debugging
# Para las primeras épocas, usamos el método fit_on_batch para poder
# aplicar curriculum learning de manera más flexible
print("\nComenzando entrenamiento con curriculum learning...")

# Si habilitar curriculum learning manual (con bucle personalizado)
USE_MANUAL_CURRICULUM = False

if USE_MANUAL_CURRICULUM:
    # Enfoque manual para curriculum learning
    for epoch in range(EPOCHS):
        # Aplicar curriculum learning para esta época
        X_curr, y_curr, decoder_inp_curr = apply_curriculum_learning(X_train, y_train, y_decoder_input_train, epoch)
        
        # Actualizar teacher forcing ratio
        progress = epoch / (EPOCHS - 1) if EPOCHS > 1 else 1.0
        global_teacher_forcing_ratio = TEACHER_FORCING_RATIO_INITIAL - progress * (TEACHER_FORCING_RATIO_INITIAL - TEACHER_FORCING_RATIO_FINAL)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} - Teacher forcing ratio: {global_teacher_forcing_ratio:.2f}")
        print(f"Curriculum learning: usando {len(X_curr)}/{len(X_train)} ejemplos")
        
        # Entrenar esta época manualmente
        # Preparar índices y barajar
        indices = np.random.permutation(len(X_curr))
        X_shuffled = X_curr[indices]
        y_shuffled = y_curr[indices]
        decoder_inp_shuffled = decoder_inp_curr[indices]
        
        # Entrenar por lotes
        progbar = tf.keras.utils.Progbar(len(X_shuffled) // BATCH_SIZE)
        losses = []
        for i in range(0, len(X_shuffled), BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, len(X_shuffled))
            batch_X = X_shuffled[i:end_idx]
            batch_y = y_shuffled[i:end_idx]
            batch_decoder_inp = decoder_inp_shuffled[i:end_idx]
            
            loss = model.train_on_batch([batch_X, batch_decoder_inp], batch_y)
            losses.append(loss[0])  # Guardar loss
            progbar.add(1, values=[('loss', loss[0]), ('accuracy', loss[1])])
        
        # Evaluar en validación
        val_loss, val_acc = model.evaluate([X_val, y_decoder_input_val], y_val, verbose=0)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        
        # Guardar mejor modelo
        if epoch == 0 or val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Saving best model with val_loss: {best_val_loss:.4f}")
            model.save('lsa_seq2seq_best_model.keras')
        
        # Early stopping
        if early_stopping_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
        if epoch > 0 and val_loss >= best_val_loss:
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0
        
        # Learning rate decay
        if lr_plateau_counter >= PATIENCE // 2:
            current_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
            new_lr = current_lr * 0.5
            if new_lr >= MIN_LR:
                tf.keras.backend.set_value(model.optimizer.learning_rate, new_lr)
                print(f"Reducing learning rate from {current_lr} to {new_lr}")
            lr_plateau_counter = 0
        if epoch > 0 and val_loss >= best_val_loss:
            lr_plateau_counter += 1
        else:
            lr_plateau_counter = 0
else:
    # Enfoque de entrenamiento estándar con callbacks
    history = model.fit(
        [X_train, y_decoder_input_train],
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([X_val, y_decoder_input_val], y_val),
        callbacks=[combined_callback, early_stopping, reduce_lr, checkpoint, nan_inf_detection],
        verbose=1
    )

# Guardar un modelo compatible para inferencia
model.save('lsa_seq2seq_model.keras')
print('Entrenamiento finalizado. Modelo guardado como lsa_seq2seq_model.keras')

# --- Guardar el tokenizer para inferencia ---
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# --- Evaluación automática con BLEU y WER ---
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from jiwer import wer
import random

print("\nEJEMPLOS DE TRADUCCIÓN (ground truth vs. predicción):")

# --- Crear modelos de inferencia adaptados a la nueva arquitectura ---

# Modelo encoder para inferencia
if USE_BIDIRECTIONAL:
    # Obtener la capa bidirectional por su nombre
    bidirectional_layer = [layer for layer in model.layers if isinstance(layer, keras.layers.Bidirectional)][0]
    
    # Si es bidireccional, capturar los estados finales y la secuencia completa para atención
    encoder_outputs_inference = encoder_outputs
    
    # Crear un modelo temporal para obtener las salidas del encoder bidireccional
    temp_model = keras.Model(encoder_inputs, bidirectional_layer.output)
    
    # Crear entradas de prueba para obtener la forma de salida
    dummy_input = np.zeros((1, MAX_SEQ_LEN, N_KEYPOINTS, N_FEATURES))
    outputs = temp_model.predict(dummy_input, verbose=0)
    
    # Ahora construimos las concatenaciones correctamente
    # Extraer estados forward y backward del encoder bidireccional
    forward_h = keras.layers.Lambda(lambda x: x[1])(bidirectional_layer.output)
    forward_c = keras.layers.Lambda(lambda x: x[2])(bidirectional_layer.output)
    backward_h = keras.layers.Lambda(lambda x: x[3])(bidirectional_layer.output)
    backward_c = keras.layers.Lambda(lambda x: x[4])(bidirectional_layer.output)
    
    # Estados concatenados para el decoder
    encoder_h = keras.layers.concatenate([forward_h, backward_h])
    encoder_c = keras.layers.concatenate([forward_c, backward_c])
    
    encoder_states_inference = [encoder_h, encoder_c]
    encoder_model = keras.Model(encoder_inputs, [encoder_outputs_inference] + encoder_states_inference)
else:
    # Si no es bidireccional, modelo más simple
    encoder_states_inference = [enc_states_forward[-1][0], enc_states_forward[-1][1]]
    encoder_model = keras.Model(encoder_inputs, [encoder_outputs] + encoder_states_inference)

# Decoder para inferencia
# Entradas: token actual + estados anteriores + salidas encoder (para atención)
decoder_inputs_inference = keras.layers.Input(shape=(1,))

# Estados iniciales para cada capa del decoder
decoder_state_inputs = []
for i in range(DECODER_LAYERS):
    # Si es bidireccional, el estado tiene el doble de unidades en la primera capa
    if i == 0 and USE_BIDIRECTIONAL:
        h_input = keras.layers.Input(shape=(DECODER_UNITS * 2,), name=f"decoder_h_input_{i+1}")
        c_input = keras.layers.Input(shape=(DECODER_UNITS * 2,), name=f"decoder_c_input_{i+1}")
    else:
        h_input = keras.layers.Input(shape=(DECODER_UNITS,), name=f"decoder_h_input_{i+1}")
        c_input = keras.layers.Input(shape=(DECODER_UNITS,), name=f"decoder_c_input_{i+1}")
    decoder_state_inputs.append([h_input, c_input])

# Input para encodings si usamos atención
if USE_ATTENTION:
    encoder_outputs_input = keras.layers.Input(shape=(MAX_SEQ_LEN, ENCODER_UNITS * (2 if USE_BIDIRECTIONAL else 1)))

# Capas de inferencia
dec_emb_inference = keras.layers.Embedding(
    input_dim=vocab_size, 
    output_dim=EMBEDDING_DIM, 
    mask_zero=True, 
    name='decoder_embedding_inference'
)(decoder_inputs_inference)

# Aplicar dropout al embedding
dec_emb_inference = keras.layers.Dropout(EMBEDDING_DROPOUT)(dec_emb_inference)

# Construir la cadena de inferencia del decoder
current_input = dec_emb_inference
decoder_outputs_inference = []
decoder_states_inference = []

# Recrear la arquitectura del decoder para inferencia
for i in range(DECODER_LAYERS):
    # Aplicar atención en la primera capa si está habilitada
    if i == 0 and USE_ATTENTION:
        # Reshape para manejar un solo token
        reshaped_decoder = keras.layers.Reshape((1, -1))(current_input)
        
        # Crear atención para inferencia
        attention_inference = BahdanauAttention(ATTENTION_UNITS)
        context_vector, attention_weights = attention_inference(decoder_state_inputs[0][0], encoder_outputs_input)
        
        # Expandir el vector de contexto para que coincida con la secuencia
        context_expanded = keras.layers.Lambda(
            lambda x: tf.expand_dims(x, 1)
        )(context_vector)
        
        # Concatenar con la entrada actual
        current_input = keras.layers.Concatenate()([reshaped_decoder, context_expanded])
    
    # Configurar LSTM para esta capa
    # Primera capa podría necesitar más unidades si viene de un encoder bidireccional
    lstm_units = DECODER_UNITS * 2 if (i == 0 and USE_BIDIRECTIONAL) else DECODER_UNITS
    lstm_layer_inference = keras.layers.LSTM(
        lstm_units if i == 0 else DECODER_UNITS,  # Primera capa puede tener más unidades
        return_sequences=True,
        return_state=True,
        dropout=0,  # No dropout en inferencia
        recurrent_dropout=0,  # No dropout en inferencia
        name=f"decoder_lstm_inference_{i+1}"
    )
    
    # Aplicar LSTM
    outputs, state_h, state_c = lstm_layer_inference(
        current_input, 
        initial_state=decoder_state_inputs[i]
    )
    
    # Guardar resultados
    decoder_outputs_inference.append(outputs)
    decoder_states_inference.append([state_h, state_c])
    current_input = outputs

# Proyectar a dimensión correcta si es necesario antes de la capa final
final_output = decoder_outputs_inference[-1]
if DECODER_LAYERS > 0 and USE_BIDIRECTIONAL and DECODER_LAYERS == 1:
    # Si solo hay una capa y es bidireccional, proyectar a la dimensión correcta
    projection_layer = keras.layers.Dense(DECODER_UNITS, activation='relu')
    final_output = projection_layer(final_output)

# Capa densa final
decoder_dense_inference = keras.layers.Dense(vocab_size, activation='softmax')
dec_outputs_inference = decoder_dense_inference(final_output)

# Definir las entradas y salidas del modelo de inferencia
if USE_ATTENTION:
    decoder_model_inputs = [decoder_inputs_inference, encoder_outputs_input]
    for states in decoder_state_inputs:
        decoder_model_inputs.extend(states)
else:
    decoder_model_inputs = [decoder_inputs_inference]
    for states in decoder_state_inputs:
        decoder_model_inputs.extend(states)

# Definir las salidas del modelo de inferencia (predicción + nuevos estados)
decoder_model_outputs = [dec_outputs_inference]
for states in decoder_states_inference:
    decoder_model_outputs.extend(states)

# Crear el modelo de decoder para inferencia
decoder_model = keras.Model(decoder_model_inputs, decoder_model_outputs)
# --- Evaluación limitada a 10 ejemplos con inferencia paso a paso ---
bleu_scores = []
wer_scores = []
indices_eval = random.sample(range(len(X_val)), min(10, len(X_val)))
def decode_sequence(input_seq):
    # Codificar la entrada con el modelo de inferencia
    if USE_ATTENTION or USE_BIDIRECTIONAL:
        # Para modelos con atención o bidireccionales
        outputs_and_states = encoder_model.predict(input_seq)
        encoder_outputs = outputs_and_states[0]  # Primer elemento son los outputs completos
        states_value = outputs_and_states[1:]    # Resto son los estados
    else:
        # Versión simple
        states_value = encoder_model.predict(input_seq)
    
    # Inicializar states para todas las capas del decoder
    decoder_states = []
    if USE_BIDIRECTIONAL:
        # Con encoder bidireccional, los estados iniciales ya están procesados
        decoder_states = [[states_value[0], states_value[1]]]
        for i in range(1, DECODER_LAYERS):
            # Para capas adicionales, iniciar con los mismos estados
            decoder_states.append([states_value[0], states_value[1]])
    else:
        # Con encoder unidireccional, usar los estados directamente
        for i in range(DECODER_LAYERS):
            if i == 0:
                # Primera capa usa estados del encoder
                decoder_states.append([states_value[0], states_value[1]])
            else:
                # Capas subsiguientes usan copia de la primera
                decoder_states.append([states_value[0], states_value[1]])
    
    # Aplanar la lista de estados para la predicción
    flat_states = []
    for state_pair in decoder_states:
        flat_states.extend(state_pair)
    
    # Inicializa secuencia objetivo con token de inicio
    target_seq = np.zeros((1, 1))
    # IMPORTANTE: El primer token debe ser <start> (no 0)
    start_index = word_index.get('<start>', 2)
    target_seq[0, 0] = start_index
    
    # Variables para almacenar los resultados
    stop_condition = False
    decoded_sentence = []
    prev_tokens = deque(maxlen=10)  # Ampliamos para mantener más contexto
    prev_bigrams = deque(maxlen=10)  # Para detectar repetición de bigramas
    prev_trigrams = deque(maxlen=10)  # Para detectar repetición de trigramas
    step = 0
    max_length = MAX_TARGET_LEN - 1  # Longitud máxima para evitar repeticiones infinitas
    
    # Parámetros para sampling con temperatura ajustados
    temperature = SAMPLING_TEMPERATURE  # Temperatura ajustada para menor aleatoriedad
    topk = SAMPLING_TOPK  # Top-k reducido para mayor precisión
    use_topk = SAMPLING_TOPK > 0  # Usar top-k sampling si el valor es > 0
    top_p = NUCLEUS_TOP_P  # Para nucleus sampling
    use_nucleus = USE_NUCLEUS_SAMPLING  # Usar nucleus sampling según configuración
    
    print(f"Temperatura de sampling: {temperature}, Top-k: {topk if use_topk else 'No'}, Nucleus: {top_p if use_nucleus else 'No'}")
    
    # Acceder al teacher forcing ratio global (para inferencia)
    global global_teacher_forcing_ratio
    print(f"Valor actual de teacher forcing ratio: {global_teacher_forcing_ratio:.2f} (solo para referencia)")
    
    # Tokens comunes en español para asegurar diversidad
    common_tokens = []
    for w in ['el', 'la', 'de', 'en', 'que', 'y', 'a', 'con', 'por', 'un', 'una', 'los', 'las']:
        if w in word_index:
            common_tokens.append(word_index[w])
    
    # Mejor manejo de tokens especiales
    pad_index = word_index.get('<pad>', 0)
    start_index = word_index.get('<start>', 2)
    end_index = word_index.get('<end>', 3)
    unk_index = word_index.get('<unk>', 1)
    special_tokens = [pad_index, start_index, end_index, unk_index]
    
    repetition_counter = {}  # Para contar repeticiones de tokens
    
    while not stop_condition:
        # Predecir token y estados
        if USE_ATTENTION:
            # Con atención: [target_seq, encoder_outputs, *flat_states]
            decoder_inputs = [target_seq, encoder_outputs] + flat_states
        else:
            # Sin atención: [target_seq, *flat_states]
            decoder_inputs = [target_seq] + flat_states
        
        # Realizar predicción
        decoder_outputs = decoder_model.predict(decoder_inputs)
        
        # Extraer probabilidades y estados
        output_tokens = decoder_outputs[0]  # Primer elemento son las probabilidades
        new_states = decoder_outputs[1:]    # Resto son los estados actualizados
        
        # Actualizar estados para la próxima iteración
        flat_states = new_states
        
        # Trabajar con las probabilidades para generar el siguiente token
        probs = output_tokens[0, 0, :].copy()  # Copiar para modificar
        
        # Aplicar temperatura para aumentar diversidad
        if temperature != 1.0:
            logits = np.log(np.clip(probs, 1e-10, 1.0)) / temperature
            exp_logits = np.exp(logits - np.max(logits))  # Evitar overflow
            probs = exp_logits / np.sum(exp_logits)
        
        # Penalización de tokens especiales
        for idx in special_tokens:
            probs[idx] *= 0.001  # Penalización fuerte para tokens especiales
        
        # Permitir <end> solo después de cierto número de tokens generados
        if step < 3:
            probs[end_index] *= 0.001  # Evitar <end> demasiado temprano
        elif step > max_length * 0.7:  # Dar más probabilidad a <end> cerca del final
            probs[end_index] *= 2.0
        
        # Penalización de repeticiones unigram
        for tok in prev_tokens:
            # Contar repeticiones para penalización exponencial
            repetition_counter[tok] = repetition_counter.get(tok, 0) + 1
            # Penalización exponencial basada en frecuencia
            repeat_penalty = 0.01 ** repetition_counter[tok] if repetition_counter[tok] > 1 else 0.1
            probs[tok] *= repeat_penalty
        
        # Penalización extra para el token más reciente (evitar repetición inmediata)
        if len(prev_tokens) > 0 and prev_tokens[-1] not in special_tokens:
            probs[prev_tokens[-1]] *= 0.01
        
        # Penalizar bigramas repetidos (evita patrones como "de la de la")
        if len(prev_tokens) >= 2:
            current_bigram = (prev_tokens[-2], prev_tokens[-1])
            if current_bigram in prev_bigrams:
                for idx in current_bigram:
                    probs[idx] *= 0.01
            prev_bigrams.append(current_bigram)
        
        # Penalizar trigramas repetidos
        if len(prev_tokens) >= 3:
            current_trigram = (prev_tokens[-3], prev_tokens[-2], prev_tokens[-1])
            if current_trigram in prev_trigrams:
                for idx in current_trigram:
                    probs[idx] *= 0.001
            prev_trigrams.append(current_trigram)
        
        # Normalizar probabilidades
        probs = probs / np.sum(probs)
        
        # Top-k sampling o Nucleus sampling para más diversidad
        if use_topk:
            # Top-k sampling: considerar solo los k tokens más probables
            top_indices = np.argsort(probs)[-topk:]
            mask = np.zeros_like(probs)
            mask[top_indices] = 1
            probs = probs * mask
            probs = probs / np.sum(probs)
        elif use_nucleus:
            # Nucleus sampling: considerar tokens hasta cubrir top_p de probabilidad
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff_idx = np.sum(cumulative_probs <= top_p) + 1
            top_indices = sorted_indices[:cutoff_idx]
            mask = np.zeros_like(probs)
            mask[top_indices] = 1
            probs = probs * mask
            probs = probs / np.sum(probs)
        
        # En casos de problemas repetitivos, forzar diversidad dando prioridad a tokens comunes
        if step > 2 and len(set(prev_tokens)) < 3:  # Si solo hay 1-2 tokens únicos, forzar diversidad
            boost_common = np.zeros_like(probs)
            for idx in common_tokens:
                if idx not in prev_tokens:  # Solo boost a tokens que no se han usado
                    boost_common[idx] = 0.1 / len(common_tokens)
            probs = probs + boost_common
            probs = probs / np.sum(probs)  # Renormalizar

        # DEBUG: Mostrar top 3 tokens con probabilidad
        if True:  # Cambiar a False para deshabilitar debug
            top_indices = np.argsort(probs)[-3:]
            top_probs = probs[top_indices]
            top_words = ['<desconocido>'] * 3
            for i, idx in enumerate(top_indices):
                for word, index in word_index.items():
                    if index == idx:
                        top_words[i] = word
                        break
            print(f"Paso {step} | Top tokens: {list(zip(top_words, top_probs))}")
        
        # Sampling con temperatura (más aleatorio que argmax)
        if temperature > 1.0 or use_topk or use_nucleus:
            # Muestreo aleatorio según las probabilidades
            sampled_token_index = np.random.choice(len(probs), p=probs)
        else:
            # Método determinístico (argmax)
            sampled_token_index = np.argmax(probs) 
        
        # Si el token seleccionado es un token especial no deseado, seleccionar alternativa
        if (step < 3 and sampled_token_index == end_index) or sampled_token_index == pad_index:
            # Seleccionar un token común alternativo
            filtered_probs = probs.copy()
            filtered_probs[end_index] = 0
            filtered_probs[pad_index] = 0
            filtered_probs = filtered_probs / np.sum(filtered_probs)
            sampled_token_index = np.random.choice(len(filtered_probs), p=filtered_probs)

        # Rastrear el token para penalizar repeticiones
        prev_tokens.append(sampled_token_index)
        
        # DEBUG: Mostrar token seleccionado
        selected_word = '<desconocido>'
        for word, index in word_index.items():
            if index == sampled_token_index:
                selected_word = word
                break
        print(f"Token seleccionado: {selected_word} (id: {sampled_token_index})")
        
        # Guardar resultado
        decoded_sentence.append(sampled_token_index)
        
        # Actualizar variables para el próximo ciclo
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        # Verificar condiciones de parada
        stop_condition = (
            sampled_token_index == end_index or  # Token <end>
            len(decoded_sentence) >= max_length    # Longitud máxima alcanzada
        )
        
        # Detección adicional de bucles: si hay 3+ repeticiones del mismo token o bigramas, forzar stop
        if len(decoded_sentence) >= 6:
            last_3_tokens = decoded_sentence[-3:]
            last_6_tokens = decoded_sentence[-6:]
            # Verificar si hay un patrón repetitivo simple (ej. a b a b a b)
            if last_6_tokens[:3] == last_6_tokens[3:] or len(set(last_3_tokens)) == 1:
                # Forzar stop y añadir <end> si hay bucle
                stop_condition = True
                if decoded_sentence[-1] != end_index:
                    decoded_sentence.append(end_index)  # Añadir <end> token
                    
        step += 1
        
    return decoded_sentence

# --- Funciones de evaluación y conversión índice-palabra ---
def convert_to_words(token_indices, word_index):
    """Convierte una secuencia de índices en palabras, excluyendo tokens especiales"""
    # Crear un mapeo inverso índice -> palabra
    index_to_word = {index: word for word, index in word_index.items()}
    
    # Tokens especiales que deben excluirse de la salida final
    special_tokens = ['<pad>', '<start>', '<end>', '<unk>']
    
    # Convertir índices a palabras, filtrando tokens especiales
    words = []
    for idx in token_indices:
        if idx in index_to_word:
            word = index_to_word[idx]
            if word not in special_tokens:
                words.append(word)
        else:
            # Si el índice no está en el mapeo, usar un placeholder
            words.append(f"[UNK-{idx}]")
    
    return words

# --- Evaluación con BLEU y WER ---
print("\n=== EVALUACIÓN CON MÉTRICAS AUTOMÁTICAS ===")
bleu_scores = []
wer_scores = []

# Seleccionar ejemplos para evaluar (máximo 10)
val_indices = random.sample(range(len(X_val)), min(10, len(X_val)))
print(f"Evaluando {len(val_indices)} ejemplos aleatorios...\n")

for i in val_indices:
    # Entrada del encoder
    enc_input = X_val[i:i+1]
    
    # Generar secuencia con el modelo actual
    decoded_indices = decode_sequence(enc_input)
    
    # Convertir índices a palabras, filtrando tokens especiales
    decoded_words = convert_to_words(decoded_indices, word_index)
    decoded_sentence = ' '.join(decoded_words)
    
    # Referencia (ground truth) limpia
    gt_raw = frases_val[i]
    gt = gt_raw.replace('<start>', '').replace('<end>', '').strip()
    
    # Calcular métricas
    bleu_score = sentence_bleu([gt.split()], decoded_words, 
                              smoothing_function=SmoothingFunction().method1)
    wer_score = wer(gt, decoded_sentence)
    
    # Mostrar resultados
    print(f"Ejemplo #{i}")
    print(f"REFERENCIA: {gt}")
    print(f"PREDICCIÓN: {decoded_sentence}")
    print(f"MÉTRICAS: BLEU={bleu_score:.3f}, WER={wer_score:.3f}")
    print("-" * 80)
    
    # Guardar métricas
    bleu_scores.append(bleu_score)
    wer_scores.append(wer_score)

# Mostrar promedio
print(f"\n=== RESULTADOS FINALES ===")
print(f"BLEU promedio ({len(bleu_scores)} ejemplos): {np.mean(bleu_scores):.3f}")
print(f"WER promedio ({len(wer_scores)} ejemplos): {np.mean(wer_scores):.3f}")
print("\nEntrenamiento y evaluación completados.")

# --- GUARDAR MODELOS Y TOKENIZER PARA INFERENCIA ---
print("\nGuardando artefactos del modelo para inferencia...")

# Definir el directorio de salida para los artefactos
artefacts_dir = "models/seq2seq_artefacts"
os.makedirs(artefacts_dir, exist_ok=True)

# Guardar el tokenizer
tokenizer_path = os.path.join(artefacts_dir, "tokenizer.pkl")
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)
print(f"Tokenizer guardado en: {tokenizer_path}")

# Guardar el modelo encoder
# Asumimos que 'encoder_model' está definido y accesible en este scope
# Si no lo está, necesitaríamos buscar dónde se define y pasarlo o re-crearlo
# con los pesos del modelo entrenado principal.
# Por ahora, asumimos que existe:
if 'encoder_model' in globals() and encoder_model is not None:
    encoder_model_path = os.path.join(artefacts_dir, "encoder_model.keras")
    encoder_model.save(encoder_model_path)
    print(f"Modelo Encoder guardado en: {encoder_model_path}")
else:
    print("ADVERTENCIA: 'encoder_model' no encontrado para guardar.")

# Guardar el modelo decoder
# Asumimos que 'decoder_model' está definido y accesible
if 'decoder_model' in globals() and decoder_model is not None:
    decoder_model_path = os.path.join(artefacts_dir, "decoder_model.keras")
    decoder_model.save(decoder_model_path)
    print(f"Modelo Decoder guardado en: {decoder_model_path}")
else:
    print("ADVERTENCIA: 'decoder_model' no encontrado para guardar.")

print("\nProceso finalizado. Los artefactos están listos para ser usados en la aplicación web.")
