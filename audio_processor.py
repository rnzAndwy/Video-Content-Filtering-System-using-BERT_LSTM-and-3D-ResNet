import os
import sys
import numpy as np
import joblib
import torch
from transformers import BertConfig, BertTokenizer
import nltk
import warnings
from pydub import AudioSegment
import whisper
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from functools import lru_cache
from transformers import TFBertModel
from difflib import SequenceMatcher
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging
import librosa
from scipy import signal
import soundfile as sf
from resource_helper import get_resource_path

# Suppress TensorFlow logging except for errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 1 = info, 2 = warning, 3 = error

# Suppress other library logs like "ms step"
tf.get_logger().setLevel(logging.ERROR)

# Suppress warnings and download NLTK data
warnings.filterwarnings("ignore")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Global variables for models
hybrid_model = None
bert_tokenizer = None
lstm_tokenizer = None
whisper_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration for profanity handling
PROFANITY_CONFIG = {
    # Set to 'bleep' or 'mute' to choose the method
    'method': 'bleep',
    # Bleep sound settings
    'bleep_frequency': 1000,  # Hz
    'bleep_volume': 0,        # dB - adjust as needed
}

def focal_loss(gamma=2., alpha=.478):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_mean(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1 + tf.keras.backend.epsilon())) - \
               tf.reduce_mean((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0 + tf.keras.backend.epsilon()))
    return focal_loss_fixed
    
def load_models(model_dir):
    global hybrid_model, bert_tokenizer, lstm_tokenizer, whisper_model
    
    # Use resource helper to get specific model files
    config_path = get_resource_path(os.path.join(model_dir, 'config.json'))
    hybrid_model_path = get_resource_path(os.path.join(model_dir, 'hybrid_bert_lstm_model123456.h5'))
    tokenizer_path = get_resource_path(os.path.join(model_dir, 'tokenizer123456.joblib'))
    
    # Log paths for debugging
    print(f"Using config path: {config_path}")
    print(f"Using model path: {hybrid_model_path}")
    print(f"Using tokenizer path: {tokenizer_path}")
    
    bert_config = BertConfig.from_pretrained(config_path)
    
    # Create a custom objects dictionary
    custom_objects = {
        'TFBertModel': TFBertModel(bert_config),
        'focal_loss_fixed': focal_loss(gamma=2., alpha=.478)
    }
    
    # Load the hybrid model with custom objects
    hybrid_model = tf.keras.models.load_model(
        hybrid_model_path,
        custom_objects=custom_objects
    )
    
    lstm_tokenizer = joblib.load(tokenizer_path)
    
    bert_base_path = os.path.dirname(config_path)
        
    # Verify paths before loading
    print(f"BERT base path: {bert_base_path}")
    print(f"Files in BERT base path: {os.listdir(bert_base_path) if os.path.exists(bert_base_path) else 'Directory not found'}")
        
    try:
        # Try to load directly from the model-specific path
        bert_tokenizer = BertTokenizer.from_pretrained(bert_base_path, local_files_only=True)
    except Exception as e:
        print(f"Error loading BERT tokenizer from {bert_base_path}: {str(e)}")
        
        # Fall back to loading from the vocab file directly
        try:
            vocab_file = get_resource_path(os.path.join(model_dir, 'vocab.txt'))
            print(f"Trying to load from vocab file: {vocab_file}")
            if os.path.exists(vocab_file):
                bert_tokenizer = BertTokenizer.from_pretrained(vocab_file, local_files_only=True)
            else:
                print(f"Vocab file not found at {vocab_file}")
                sys.exit(1)
        except Exception as e2:
            print(f"Error loading BERT tokenizer from vocab file: {str(e2)}")
            sys.exit(1)
    
    try:
        whisper_model = whisper.load_model("medium", device=device)
    except Exception as e:
        print(f"Error loading Whisper model: {str(e)}")
        sys.exit(1)

@lru_cache(maxsize=1000)
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens)

def create_profanity_filter_net(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        words = [word.strip().lower() for word in f.readlines()]
    return set(word for word in words if word)

def is_false_positive(word, context):
    false_positives = {
        'cock': ['clock', 'cock-a-doodle-doo'],
        'dick': ['thick', 'stick', 'trick', 'click'],
        'hell': ['hello', 'shell', 'spell'],
        'damn': ['dam', 'damp', 'damn good'],
        'bitch': ['beech','bich'],
        'shit': ['sheet','sheath']
    }
    
    for profane, safe_words in false_positives.items():
        if word.lower() == profane:
            for safe_word in safe_words:
                if safe_word.lower() in context.lower():
                    return True
    return False

def is_profane(word, profanity_net):
    return word.lower() in profanity_net

def slice_and_check_profanity(word, profanity_net):
    for i in range(1, len(word)):
        left_slice = word[:i]
        right_slice = word[i:]
        if is_profane(left_slice, profanity_net) and is_profane(right_slice, profanity_net):
            return True
    return False

def detect_ambiguous_profanity(word, profanity_net):
    def word_similarity(word1, word2):
        return SequenceMatcher(None, word1, word2).ratio()

    def is_structurally_similar(word, profane_word, threshold=0.8):
        return word_similarity(word, profane_word) >= threshold

    # Check if the whole word is in the profanity list
    if word.lower() in profanity_net:
        return True

    # Split the word into segments
    segments = []
    for i in range(1, len(word)):
        left_slice = word[:i]
        right_slice = word[i:]
        segments.append((left_slice, right_slice))

    for left_slice, right_slice in segments:
        # Check if any segment is in the profanity list
        if is_profane(left_slice, profanity_net) or is_profane(right_slice, profanity_net):
            # If a segment is found in the profanity list, check if the whole word
            # is structurally similar to any word in the profanity list
            for profane_word in profanity_net:
                if is_structurally_similar(word, profane_word):
                    return True
            
            # If no structural similarity is found, it's likely a false positive
            return False

    return False

def is_above_threshold(prediction, threshold=0.5):
    return prediction >= threshold

def process_word(word_info, lstm_tokenizer, bert_tokenizer, max_length_lstm, max_length_bert, profanity_net, context):
    word = preprocess_text(word_info["word"])
    start_time = word_info["start"]
    end_time = word_info["end"]
    
    x_lstm_seq = lstm_tokenizer.texts_to_sequences([word])
    x_lstm = pad_sequences(x_lstm_seq, maxlen=max_length_lstm, padding='post')
    
    x_bert = bert_tokenizer(
        word,
        add_special_tokens=True,
        max_length=max_length_bert,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    
    prediction = hybrid_model.predict([x_lstm, x_bert['input_ids'], x_bert['attention_mask']]).flatten()[0]
    
    is_profane_word = False
    
    if is_above_threshold(prediction):
        is_profane_word = detect_ambiguous_profanity(word, profanity_net)
    
    if is_profane_word and is_false_positive(word, context):
        is_profane_word = False
    
    return (start_time, end_time, word, is_profane_word)

def normalize_audio(audio_file):
    audio, sr = librosa.load(audio_file, sr=None)
    pre_emphasis = 0.97
    emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
    target_rms = 0.1
    rms = np.sqrt(np.mean(emphasized_audio**2))
    gain = target_rms / (rms + 1e-6)
    normalized_audio = emphasized_audio * gain

    def compress_dynamic_range(x, threshold, ratio, attack, release):
        db = 20 * np.log10(np.abs(x) + 1e-6)
        gain_reduction = np.maximum(0, db - threshold) * (1 - 1/ratio)
        coeff_a = np.exp(-1 / (sr * attack))
        coeff_r = np.exp(-1 / (sr * release))
        gr_smooth = np.zeros_like(gain_reduction)
        for i in range(1, len(gr_smooth)):
            if gain_reduction[i] > gr_smooth[i-1]:
                gr_smooth[i] = coeff_a * gr_smooth[i-1] + (1 - coeff_a) * gain_reduction[i]
            else:
                gr_smooth[i] = coeff_r * gr_smooth[i-1] + (1 - coeff_r) * gain_reduction[i]
        return x * 10**(-gr_smooth/20)

    compressed_audio = compress_dynamic_range(normalized_audio, -20, 4, 0.005, 0.05)
    nyquist = 0.5 * sr
    cutoff = 80 / nyquist
    b, a = signal.butter(6, cutoff, btype='highpass')
    filtered_audio = signal.filtfilt(b, a, compressed_audio)
    final_audio = np.clip(filtered_audio, -1, 1)
    return final_audio, sr

def detect_profanity_in_audio(audio_file, progress_callback, profanity_net):
    normalized_audio, sr = normalize_audio(audio_file)
    max_length_lstm = 150
    max_length_bert = 128

    temp_file = "temp_normalized_audio.wav"
    sf.write(temp_file, normalized_audio, sr)
    result = whisper_model.transcribe(temp_file, word_timestamps=True)
    os.remove(temp_file)

    profanity_words = []
    total_words = sum(len(segment["words"]) for segment in result["segments"])
    processed_words = 0

    for segment in result["segments"]:
        context = " ".join(word_info["word"] for word_info in segment["words"])
        for word_info in segment["words"]:
            processed_word = process_word(
                word_info, lstm_tokenizer, bert_tokenizer,
                max_length_lstm, max_length_bert, profanity_net, context
            )
            if processed_word[3]:  # If the word is profane
                profanity_words.append(processed_word[:3])  # Append (start_time, end_time, word)

            processed_words += 1
            progress = (processed_words / total_words) * 100
            progress_callback(f"Processing words: {progress:.2f}%")

    return profanity_words

def generate_bleep_sound(duration_ms, sample_rate=44100):
    """Generate a bleep sound with the specified duration."""
    frequency = PROFANITY_CONFIG['bleep_frequency']
    volume = PROFANITY_CONFIG['bleep_volume']
    
    # Generate sine wave for the bleep
    samples = np.sin(2 * np.pi * frequency * np.arange(int(duration_ms * sample_rate / 1000)) / sample_rate)
    
    # Convert to 16-bit PCM
    samples = (samples * 32767).astype(np.int16)
    
    # Create an AudioSegment from the raw PCM data
    bleep = AudioSegment(
        samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit
        channels=1
    )
    
    # Adjust volume (dB)
    bleep = bleep + volume
    
    return bleep

def filter_profanity_from_audio(audio_file, profanity_words, output_file):
    audio = AudioSegment.from_file(audio_file)
    duration_ms = len(audio)
    
    # Choose method based on configuration
    if PROFANITY_CONFIG['method'] == 'mute':
        # Muting method (original code)
        filtered_audio = AudioSegment.silent(duration=duration_ms)
        
        non_profanity_segments = []
        last_end = 0
        for start_time, end_time, _ in sorted(profanity_words):
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            start_ms = max(0, start_ms - 100)
            end_ms = min(duration_ms, end_ms + 100)
            
            if start_ms > last_end:
                non_profanity_segments.append((last_end, start_ms))
            last_end = end_ms
        
        if last_end < duration_ms:
            non_profanity_segments.append((last_end, duration_ms))
        
        for start, end in non_profanity_segments:
            filtered_audio = filtered_audio.overlay(audio[start:end], position=start)
            
    else:  # 'bleep' method
        # Start with the original audio
        filtered_audio = audio
        
        # Sort profanity words by start time (to process them in order)
        for start_time, end_time, _ in sorted(profanity_words, reverse=True):
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            # Add a small padding to make sure we completely cover the profanity
            start_ms = max(0, start_ms - 100)
            end_ms = min(duration_ms, end_ms + 100)
            
            # Generate a bleep of the correct duration
            bleep_duration = end_ms - start_ms
            bleep = generate_bleep_sound(bleep_duration)
            
            # Replace the profanity with the bleep
            filtered_audio = filtered_audio[:start_ms] + bleep + filtered_audio[end_ms:]
    
    filtered_audio.export(output_file, format="mp3")

def process_audio(audio_file, output_file, progress_callback, profanity_file):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = get_resource_path("BERT_LSTM")
        
        if not os.path.exists(model_dir):
            return {"error": f"Model directory not found at {model_dir}"}
        
        if not os.path.exists(profanity_file):
            return {"error": f"Profanity file not found at {profanity_file}"}
        
        progress_callback("Loading models")
        load_models(model_dir)
        
        progress_callback("Loading profanity list")
        profanity_net = create_profanity_filter_net(profanity_file)
        
        progress_callback("Detecting profanity in audio")
        profanity_words = detect_profanity_in_audio(audio_file, progress_callback, profanity_net)
        
        progress_callback(f"Filtering profanity from audio using {PROFANITY_CONFIG['method']} method")
        filter_profanity_from_audio(audio_file, profanity_words, output_file)
        
        return {
            "success": True,
            "filtered_audio": os.path.basename(output_file),
            "profanity_count": len(profanity_words),
            "filter_method": PROFANITY_CONFIG['method']
        }
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        return {"error": f"An unexpected error occurred: {str(e)}\n\nTraceback:\n{error_traceback}"}