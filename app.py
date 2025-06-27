from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import os
import pandas as pd
from werkzeug.utils import secure_filename
import tensorflow as tf
import pickle
import numpy as np # Necesario para algunas operaciones si ocurren aquí

# Habilitar deserialización insegura para capas Lambda (por si acaso algún subcomponente lo necesita)
tf.keras.config.enable_unsafe_deserialization()

# Importar la clase de atención y las NUEVAS funciones de construcción de modelos
from transcribe_video import (
    BahdanauAttention, # Ya está serializable
    transcribe_video_with_seq2seq,
    # Constantes que podríamos necesitar si no se pasan explícitamente
    TOKENIZER_PATH as TV_TOKENIZER_PATH, 
    ENCODER_MODEL_PATH as TV_ENCODER_MODEL_PATH,
    DECODER_MODEL_PATH as TV_DECODER_MODEL_PATH,
    NUM_WORDS as TV_NUM_WORDS # Para vocab_size
)

app = Flask(__name__)

# --- Rutas a los artefactos del modelo Seq2Seq (usando las de transcribe_video) ---
SEQ2SEQ_ARTEFACTS_DIR = "models/seq2seq_artefacts" # O usar os.path.dirname(TV_TOKENIZER_PATH)
TOKENIZER_PATH = TV_TOKENIZER_PATH
ENCODER_MODEL_PATH = TV_ENCODER_MODEL_PATH
DECODER_MODEL_PATH = TV_DECODER_MODEL_PATH

# --- Cargar modelos y tokenizer una vez al iniciar la app ---
encoder_model_loaded = None
decoder_model_loaded = None
tokenizer_loaded = None
initialization_error = None

try:
    print("Cargando tokenizer...")
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer_loaded = pickle.load(handle)
    print(f"Tokenizer cargado. Word index size: {len(tokenizer_loaded.word_index)}")
    
    # El vocab_size para la capa Embedding del decoder debe coincidir con el usado en el entrenamiento.
    # Generalmente es min(num_words_config, len(word_index) + 1)
    # Usaremos TV_NUM_WORDS que es el num_words del config de entrenamiento.
    # actual_vocab_size = min(TV_NUM_WORDS, len(tokenizer_loaded.word_index) + 1) # Ya no es necesario aquí
    # print(f"Vocab size para el decoder: {actual_vocab_size}")

    print("Cargando modelo Encoder Seq2Seq...")
    # Cargar modelo directamente
    encoder_model_loaded = tf.keras.models.load_model(ENCODER_MODEL_PATH, custom_objects={'BahdanauAttention': BahdanauAttention})
    print("Encoder model cargado.")
    # encoder_model_loaded.summary() # Descomentar para depurar estructura

    print("Cargando modelo Decoder Seq2Seq...")
    # Cargar modelo directamente
    decoder_model_loaded = tf.keras.models.load_model(DECODER_MODEL_PATH, custom_objects={'BahdanauAttention': BahdanauAttention})
    print("Decoder model cargado.")
    # decoder_model_loaded.summary() # Descomentar para depurar estructura
    
    print("Modelos Seq2Seq y tokenizer cargados exitosamente.")

except FileNotFoundError as e:
    initialization_error = f"Error de archivo no encontrado durante la inicialización: {e}. Asegúrate de que los artefactos del modelo ({ENCODER_MODEL_PATH}, {DECODER_MODEL_PATH}, {TOKENIZER_PATH}) existen."
    print(initialization_error)
except Exception as e:
    initialization_error = f"Error durante la carga de modelos/tokenizer: {e}. Revisa las trazas del servidor para más detalles."
    print(initialization_error)
    # Podrías querer imprimir la traza completa aquí para depuración:
    import traceback
    traceback.print_exc()


UPLOAD_FOLDER = 'data/clips/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# NUEVA RUTA para servir archivos desde UPLOAD_FOLDER:
@app.route('/data_files/<path:filename>') # Cambiado nombre de ruta para evitar conflicto si 'data' es un blueprint
def serve_uploaded_file(filename):
    # Servir desde la ruta absoluta es más seguro con send_from_directory
    # UPLOAD_FOLDER ya es 'data/clips/'
    return send_from_directory(os.path.abspath(app.config['UPLOAD_FOLDER']), filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    video_url_for_template = None # Renombrado para claridad
    subtitles = None
    srt_download = None
    error_message = None
    transcription_text = None # Variable para la transcripción de la IA

    if initialization_error: # Si hubo un error al cargar modelos globalmente
        return render_template('index.html', error_message=initialization_error, transcription=transcription_text)

    if request.method == 'POST':
        if 'video' not in request.files:
            return jsonify({'error': 'No se encontró el archivo de video'}), 400
        
        video = request.files['video']
        if video.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

        if video:
            filename = secure_filename(video.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video.save(save_path)
            
            # Generar la URL usando url_for para la nueva ruta
            video_url_for_template = url_for('serve_uploaded_file', filename=filename)

            # Comprobar si los modelos se cargaron correctamente
            if encoder_model_loaded is None or decoder_model_loaded is None or tokenizer_loaded is None:
                error_message = "Error: Modelos Seq2Seq o tokenizer no cargados correctamente al inicio de la aplicación."
                # Este error no debería ocurrir si initialization_error no se disparó, pero es una doble comprobación.
                return render_template('index.html', video_url=video_url_for_template, subtitles=subtitles, srt_download=srt_download, error_message=error_message, transcription=transcription_text)

            try:
                print(f"Transcribiendo video: {save_path}")
                transcribed_text = transcribe_video_with_seq2seq(
                    save_path, 
                    encoder_model_loaded, 
                    decoder_model_loaded, 
                    tokenizer_loaded
                )
                print(f"Texto transcrito: {transcribed_text}")
                transcription_text = transcribed_text # Asignar a la variable que se pasará a la plantilla
                
                # Generar contenido SRT simple (un solo bloque de tiempo para toda la transcripción)
                # Puedes hacerlo más sofisticado si tienes timestamps por palabra/frase
                srt_content = f"1\n00:00:00,000 --> 00:00:10,000\n{transcribed_text}\n" # Ejemplo de 10s
                
                subtitles = srt_content # Para mostrar en la página
                
                # Guardar archivo SRT para descarga
                srt_filename = os.path.splitext(filename)[0] + ".srt"
                srt_path = os.path.join(app.config['UPLOAD_FOLDER'], srt_filename)
                with open(srt_path, "w", encoding='utf-8') as f_srt:
                    f_srt.write(srt_content)
                srt_download = srt_path # Ruta para el enlace de descarga

            except Exception as e:
                print(f"Error durante la transcripción del video {filename}: {e}")
                import traceback
                traceback.print_exc()
                error_message = f"Error al procesar el video: {e}"

    return render_template('index.html', video_url=video_url_for_template, subtitles=subtitles, srt_download=srt_download, error_message=error_message, transcription=transcription_text)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)