import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras # Asegurarse de que keras se importa desde tf
import pickle
from collections import deque
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction # Podría no ser necesaria aquí pero sí para decode_sequence
import os

from extract_keypoints import extract_keypoints_from_video

# --- Rutas a los artefactos del modelo Seq2Seq ---
SEQ2SEQ_ARTEFACTS_DIR = "models/seq2seq_artefacts"
ENCODER_MODEL_PATH = os.path.join(SEQ2SEQ_ARTEFACTS_DIR, "encoder_model.keras")
DECODER_MODEL_PATH = os.path.join(SEQ2SEQ_ARTEFACTS_DIR, "decoder_model.keras")
TOKENIZER_PATH = os.path.join(SEQ2SEQ_ARTEFACTS_DIR, "tokenizer.pkl")
# --- (Fin Rutas Seq2Seq) ---

# Mantener LABELS_PATH y NORMALIZATION_PATH si se usan para algo más,
# o si la segmentación aún los necesita de alguna forma.
# MODEL_PATH original ya no se usará para el modelo seq2seq.
LABELS_PATH = "labels.txt"
TRANSCRIPTION_PATH = "transcription.txt"
NORMALIZATION_PATH = "normalization.txt"

# --- Constantes adaptadas de train_seq2seq.py para la construcción de modelos y preprocesamiento ---
MAX_SEQ_LEN_KEYPOINTS = 30
N_KEYPOINTS = 33
N_FEATURES = 3    # x, y, confidence (o x, y, z según el preproc final)
MAX_TARGET_LEN_TEXT = 30
NUM_WORDS = 5000 # Vocab size para el tokenizer, el real puede ser menor

# Arquitectura del modelo (deben coincidir con train_seq2seq.py)
EMBEDDING_DIM = 128
ENCODER_UNITS = 128
DECODER_UNITS = 128
ATTENTION_UNITS = 128 # Usado en BahdanauAttention
# ENCODER_LAYERS y DECODER_LAYERS se infieren de la estructura de los modelos guardados.
USE_ATTENTION = True # Determina si se usa atención en el decoder y en decode_sequence
USE_BIDIRECTIONAL = True # Importante para la forma de los estados del encoder
EMBEDDING_DROPOUT = 0.2 # Usado en la construcción del decoder de inferencia

# Parámetros de muestreo (usados en decode_sequence_from_models)
SAMPLING_TEMPERATURE = 1.0
SAMPLING_TOPK = 5
USE_NUCLEUS_SAMPLING = True
NUCLEUS_TOP_P = 0.8

x_min = 0.0 
y_min = 0.0
x_max = 1.0
y_max = 1.0
WINDOW_SIZE = 60 # Esto parece ser de la lógica de clasificación anterior
STEP_SIZE = 30   # Esto parece ser de la lógica de clasificación anterior

# Definición de BahdanauAttention (debe ser idéntica a la de train_seq2seq.py)
@tf.keras.utils.register_keras_serializable()
class BahdanauAttention(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        # Pesos definidos en build o __init__ para compatibilidad
        self.W1 = keras.layers.Dense(units, use_bias=False, name='attention_W1')
        self.W2 = keras.layers.Dense(units, use_bias=False, name='attention_W2')
        self.V = keras.layers.Dense(1, use_bias=False, name='attention_V')

    def build(self, input_shape):
        # input_shape es una tupla de formas: (query_shape, values_shape)
        # query_shape: (batch_size, query_features)
        # values_shape: (batch_size, seq_len_values, values_features)
        # La forma de W1 debe ser (values_features, units)
        # La forma de W2 debe ser (query_features, units)
        # La forma de V debe ser (units, 1)
        # Esto ya está manejado por Dense si se crean en __init__ con las unidades correctas.
        # Si se crean aquí, se necesitaría input_shape[-1] para las dimensiones.
        super(BahdanauAttention, self).build(input_shape) 

    def call(self, query, values):
        # query shape == (batch_size, hidden_size)
        # values shape == (batch_size, max_len, hidden_size)
        query_expanded = tf.expand_dims(query, 1) # (batch_size, 1, hidden_size)
        
        # score shape == (batch_size, max_len, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_len, units)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query_expanded)))
        
        # attention_weights shape == (batch_size, max_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = tf.reduce_sum(attention_weights * values, axis=1)
        
        return context_vector, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

# --- Funciones para construir modelos de inferencia ---
# Estas funciones ya no son necesarias, cargaremos los modelos directamente.
# def build_inference_encoder_model():
#     ... (código eliminado) ...
# def build_inference_decoder_model(vocab_size_param):
#     ... (código eliminado) ...

def load_normalization_params(path):
    with open(path, "r", encoding="utf-8") as f:
        vals = f.read().strip().split()
        return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])

# Esta normalización de keypoints es la de la CLASIFICACIÓN.
# La normalización de train_seq2seq.py es diferente (normalize_keypoints y luego BatchNormalization en el modelo).
# Deberíamos usar la misma lógica de preprocesamiento de keypoints que en train_seq2seq.py.
# Por ahora, la comentaré y usaremos la de train_seq2seq.py si es necesario.
# def normalize_keypoints_original(keypoints):
#     keypoints = keypoints.copy()
#     keypoints[..., 0] = (keypoints[..., 0] - x_min) / (x_max - x_min + 1e-8)
#     keypoints[..., 1] = (keypoints[..., 1] - y_min) / (y_max - y_min + 1e-8)
#     keypoints = np.nan_to_num(keypoints, nan=0.0)
#     return keypoints

# Nueva función de normalización de keypoints (adaptada de train_seq2seq.py)
def normalize_keypoints_seq2seq(keypoints_data):
    norm_keypoints = keypoints_data.copy()
    for i in range(norm_keypoints.shape[0]): # Iterar sobre frames
        confidence = norm_keypoints[i, :, 2]
        valid_mask = confidence > 0.3
        if np.sum(valid_mask) < 5:
            continue
        valid_kps = norm_keypoints[i, valid_mask, :2]
        
        center_point = np.array([0.0, 0.0]) # Inicializar center_point
        if valid_mask[11] and valid_mask[12]:
            center_point = (norm_keypoints[i, 11, :2] + norm_keypoints[i, 12, :2]) / 2
        else:
            center_point = np.mean(valid_kps, axis=0)
            
        norm_keypoints[i, :, 0] = norm_keypoints[i, :, 0] - center_point[0]
        norm_keypoints[i, :, 1] = norm_keypoints[i, :, 1] - center_point[1]
        
        distances = np.sqrt(np.sum(valid_kps**2, axis=1)) # Re-calcular distances sobre valid_kps centrados
        scale_factor = np.mean(distances) + 1e-6
        
        norm_keypoints[i, :, 0] = norm_keypoints[i, :, 0] / scale_factor
        norm_keypoints[i, :, 1] = norm_keypoints[i, :, 1] / scale_factor
    
    # En train_seq2seq, también hay una normalización Z-score y clipping.
    # Por simplicidad inicial, solo aplicamos la normalización espacial.
    # Si los resultados no son buenos, se puede añadir la normalización Z-score aquí.
    norm_keypoints = np.nan_to_num(norm_keypoints, nan=0.0, posinf=0.0, neginf=0.0)
    return norm_keypoints


def convert_seq_to_words(token_indices, tokenizer_map):
    index_to_word = {index: word for word, index in tokenizer_map.items()}
    special_tokens = ['<pad>', '<start>', '<end>', '<unk>']
    words = []
    for idx in token_indices:
        if idx in index_to_word:
            word = index_to_word[idx]
            if word not in special_tokens:
                words.append(word)
        # else: # Opcional: manejar tokens desconocidos si es necesario
        #     words.append(f"[UNK-{idx}]") 
    return words

# Adaptación de la función decode_sequence de train_seq2seq.py
def decode_sequence_from_models(input_seq_keypoints, encoder_m, decoder_m, tokenizer_w_index, max_len_text):
    # Esta función es una adaptación. Necesita acceso a:
    # encoder_model, decoder_model, tokenizer.word_index, MAX_TARGET_LEN_TEXT
    # SAMPLING_TEMPERATURE, SAMPLING_TOPK, USE_NUCLEUS_SAMPLING, NUCLEUS_TOP_P
    # Y las constantes USE_ATTENTION, USE_BIDIRECTIONAL (asumidas de la estructura del modelo guardado)
    
    # Asumimos que USE_ATTENTION y USE_BIDIRECTIONAL son ciertas por la estructura de train_seq2seq
    # Estas podrían necesitar ser pasadas o inferidas si los modelos pueden variar.
    
    encoder_outputs_inf, state_h_enc, state_c_enc = encoder_m.predict(input_seq_keypoints, verbose=0)
    decoder_states_inf = [state_h_enc, state_c_enc]

    target_seq = np.zeros((1, 1))
    start_token_idx = tokenizer_w_index.get('<start>', 2) # Usar 2 como default si no existe
    target_seq[0, 0] = start_token_idx

    stop_condition = False
    decoded_sentence_tokens = []
    
    # Para penalización de repeticiones (simplificado de train_seq2seq)
    prev_tokens_q = deque(maxlen=10)

    step = 0
    while not stop_condition:
        decoder_model_inputs = [target_seq] + decoder_states_inf
        # Asumiendo que la capa de atención se llama 'decoder_attention' y está en el decoder_model
        # o que el decoder_model está construido para tomar encoder_outputs_inf si hay atención
        # La estructura de train_seq2seq.py sugiere que encoder_outputs_inf se pasa si hay atención.
        # Aquí asumimos que el decoder_model guardado ya tiene la estructura correcta para tomar encoder_outputs_inf si es necesario.
        # La creación del decoder_model en train_seq2seq.py es:
        # decoder_model_inputs_list = [decoder_inputs_inf] + decoder_states_inputs_inf
        # if USE_ATTENTION: decoder_model_inputs_list.append(encoder_outputs_inf_input)

        # Para determinar si pasar encoder_outputs_inf, necesitamos saber si el decoder_model lo espera.
        # Esto se puede inferir del número de inputs del decoder_model.
        if len(decoder_m.inputs) > 3: # [target_seq, state_h, state_c, encoder_outputs_inf (opcional)]
             decoder_model_inputs.append(encoder_outputs_inf)

        output_tokens, h_state_dec, c_state_dec = decoder_m.predict(decoder_model_inputs, verbose=0)
        decoder_states_inf = [h_state_dec, c_state_dec]

        probs = output_tokens[0, 0, :].copy()

        # Lógica de sampling (simplificada, se puede expandir como en train_seq2seq)
        if SAMPLING_TEMPERATURE > 0.0: # Evitar división por cero si temperature es 0
            logits = np.log(np.clip(probs, 1e-10, 1.0)) / SAMPLING_TEMPERATURE
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

        # Penalización de repeticiones (simple)
        for token_id in prev_tokens_q:
            if token_id < len(probs):
                probs[token_id] *= 0.1 
        if len(prev_tokens_q) > 0 and prev_tokens_q[-1] < len(probs):
             probs[prev_tokens_q[-1]] *= 0.01
        
        # Top-k o Nucleus sampling (adaptado)
        if SAMPLING_TOPK > 0 and not USE_NUCLEUS_SAMPLING:
            top_indices = np.argsort(probs)[-SAMPLING_TOPK:]
            mask = np.zeros_like(probs)
            mask[top_indices] = 1
            probs = probs * mask
            probs_sum = np.sum(probs)
            if probs_sum > 1e-6 : probs = probs / probs_sum
            else: probs = np.ones_like(probs) / len(probs) # Fallback a uniforme si todo es cero
        elif USE_NUCLEUS_SAMPLING:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff_idx = np.sum(cumulative_probs <= NUCLEUS_TOP_P)
            if cutoff_idx == 0 and len(sorted_indices) > 0 : cutoff_idx = 1 # Asegurar al menos un token
            top_indices = sorted_indices[:cutoff_idx]
            mask = np.zeros_like(probs)
            if len(top_indices) > 0: # Verificar que top_indices no esté vacío
                 mask[top_indices] = 1
                 probs = probs * mask
                 probs_sum = np.sum(probs)
                 if probs_sum > 1e-6 : probs = probs / probs_sum
                 else: probs = np.ones_like(probs) / len(probs)
            else: # Fallback si top_indices está vacío
                 probs = np.ones_like(probs) / len(probs)


        if SAMPLING_TEMPERATURE > 0.0 or SAMPLING_TOPK > 0 or USE_NUCLEUS_SAMPLING:
             sampled_token_index = np.random.choice(len(probs), p=probs)
        else:
             sampled_token_index = np.argmax(probs)
        
        end_token_idx = tokenizer_w_index.get('<end>', 3) # Usar 3 como default
        pad_token_idx = tokenizer_w_index.get('<pad>', 0)

        if sampled_token_index == end_token_idx or sampled_token_index == pad_token_idx or len(decoded_sentence_tokens) >= max_len_text -1 :
            stop_condition = True
        else:
            decoded_sentence_tokens.append(sampled_token_index)
            prev_tokens_q.append(sampled_token_index)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        step += 1
        
    return decoded_sentence_tokens

# Esta es la función que se llamará desde app.py
def transcribe_video_with_seq2seq(video_path, encoder_m, decoder_m, tokenizer_obj):
    # 1. Extraer keypoints del video completo (o por segmentos si es muy largo)
    #    La lógica de train_seq2seq.py procesa clips completos hasta MAX_SEQ_LEN.
    #    Si el video es más largo, necesitará segmentación.
    #    Por ahora, asumimos que el video se ajusta o se procesa como un solo clip.
    
    all_keypoints = extract_keypoints_from_video(video_path, max_frames=MAX_SEQ_LEN_KEYPOINTS) # usa la función original
    
    if all_keypoints is None or all_keypoints.shape[0] == 0:
        return "No se pudieron extraer keypoints del video."

    # 2. Preprocesar keypoints (normalización, padding/truncado)
    #    Usar la normalización espacial de train_seq2seq
    #    Asegurarse de que solo se pasen x, y, confidence (o lo que espere el encoder)
    #    La función load_keypoints en train_seq2seq hace: kp = kp[:, :, [0, 1, 3]]
    #    extract_keypoints_from_video devuelve (frames, num_keypoints, 4) (x, y, z, visibility)
    #    Necesitamos seleccionar las características correctas y reordenar si es necesario.
    #    El modelo seq2seq espera (frames, N_KEYPOINTS, N_FEATURES=3) (x, y, confidence)
    
    # Asumiendo que extract_keypoints_from_video devuelve x,y,z,visibility
    # y N_FEATURES=3 se refiere a x,y,confidence (visibility)
    # Seleccionar x, y, visibility (índices 0, 1, 3)
    processed_kps = all_keypoints[:, :, [0, 1, 3]] 
    
    # Aplicar normalización espacial
    processed_kps = normalize_keypoints_seq2seq(processed_kps) # Usa la nueva normalización

    # Padding o truncado a MAX_SEQ_LEN_KEYPOINTS
    seq_kps = np.zeros((MAX_SEQ_LEN_KEYPOINTS, N_KEYPOINTS, N_FEATURES), dtype=np.float32)
    length = min(len(processed_kps), MAX_SEQ_LEN_KEYPOINTS)
    seq_kps[:length] = processed_kps[:length]
    
    input_for_encoder = np.expand_dims(seq_kps, axis=0) # Añadir dimensión de batch

    # 3. Decodificar usando encoder y decoder
    predicted_tokens = decode_sequence_from_models(
        input_for_encoder, 
        encoder_m, 
        decoder_m, 
        tokenizer_obj.word_index, 
        MAX_TARGET_LEN_TEXT
    )

    # 4. Convertir tokens a palabras
    transcribed_text = " ".join(convert_seq_to_words(predicted_tokens, tokenizer_obj.word_index))
    
    return transcribed_text


# La función segment_and_transcribe_video original es para el modelo de CLASIFICACIÓN.
# La dejamos aquí por si se necesita para alguna otra cosa, pero app.py usará la nueva.
def segment_and_transcribe_video_classification(video_path, model, labels_list): # Renombrada
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return "Error al abrir el video."
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    predictions = []
    # WINDOW_SIZE y STEP_SIZE son de la lógica de clasificación
    # Para seq2seq, normalmente se procesa una secuencia más larga o el clip completo
    # Esta segmentación puede no ser la ideal para un modelo seq2seq
    
    # Por ahora, para simplificar y probar, procesaremos el video en una sola pasada
    # usando los primeros MAX_SEQ_LEN_KEYPOINTS frames.
    # Si se requiere segmentación para videos largos, esta parte necesitará más trabajo.
    
    keypoints_full = extract_keypoints_from_video(video_path, max_frames=WINDOW_SIZE) # Usa WINDOW_SIZE de la clasificación
    
    if keypoints_full is None or keypoints_full.shape[0] == 0:
         return "No se pudieron extraer keypoints."

    # Aquí se usaba la normalización original para el modelo de clasificación
    # keypoints_norm = normalize_keypoints_original(keypoints_full)
    # Esta parte se omite ya que la nueva función de transcripción se encarga del preproc.

    # El resto de esta función es específico para clasificación
    # keypoints_padded = np.zeros((WINDOW_SIZE, 33, 4)) 
    # current_len = min(keypoints_norm.shape[0], WINDOW_SIZE)
    # keypoints_padded[:current_len] = keypoints_norm[:current_len]
    # keypoints_expanded = np.expand_dims(keypoints_padded, axis=0)
    
    # pred = model.predict(keypoints_expanded, verbose=0)
    # idx = np.argmax(pred)
    # predictions.append(labels_list[idx]) # Asume que labels_list es una lista de strings
    
    # Lo anterior es incorrecto para seq2seq.
    # La transcripción real se hará en transcribe_video_with_seq2seq
    
    # Esta función, si se mantiene, debería devolver algo como "Lógica de clasificación no implementada aquí".
    return "Esta función es para clasificación y no debe usarse con el modelo seq2seq."


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python transcribe_video.py <video_path>")
        sys.exit(1)
    
    video_path_arg = sys.argv[1]

    print("Cargando modelos Seq2Seq y tokenizer...")
    custom_objects = {'BahdanauAttention': BahdanauAttention}
    try:
        # Cargar modelos guardados directamente
        encoder_model_loaded = tf.keras.models.load_model(ENCODER_MODEL_PATH, custom_objects=custom_objects)
        decoder_model_loaded = tf.keras.models.load_model(DECODER_MODEL_PATH, custom_objects=custom_objects)
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer_loaded = pickle.load(handle)
        print("Modelos y tokenizer Seq2Seq cargados correctamente.")
    except Exception as e:
        print(f"Error al cargar los modelos Seq2Seq o el tokenizer: {e}")
        sys.exit(1)

    print(f"Procesando video con modelo Seq2Seq: {video_path_arg}")
    transcription_result = transcribe_video_with_seq2seq(
        video_path_arg, 
        encoder_model_loaded, 
        decoder_model_loaded, 
        tokenizer_loaded
    )
    
    with open(TRANSCRIPTION_PATH, "w", encoding="utf-8") as f:
        f.write(transcription_result)
    print(f"Transcripción (Seq2Seq) guardada en {TRANSCRIPTION_PATH}:")
    print(transcription_result)
