

import joblib

# Load vectorizer
tfidf = joblib.load("vectorizer/tfidf.pkl")


# Load models
svm_model = joblib.load("models/svm_stress_classifier_final.pkl")
lr_model = joblib.load("models/logistic_regression.pkl")

# Map numeric labels to human readable classes
# NOTE: This assumes the models were trained with labels: 0=Low, 1=Medium, 2=High
LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}


def _to_class_name(pred):
    return LABEL_MAP.get(int(pred), str(pred))


def predict_svm(text):
    X = tfidf.transform([text])
    pred = svm_model.predict(X)[0]
    return _to_class_name(pred)


def predict_lr(text):
    X = tfidf.transform([text])
    pred = lr_model.predict(X)[0]
    return _to_class_name(pred)


# -----------------------------
# Transformer models (DistilBERT + MentalBERT)
# -----------------------------
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    _DISTILBERT_DIR = "models/distilbert_stress_model"
    _MENTALBERT_DIR = "models/mentalbert-finetuned"

    _tokenizer_distil = AutoTokenizer.from_pretrained(_DISTILBERT_DIR)
    _model_distil = AutoModelForSequenceClassification.from_pretrained(_DISTILBERT_DIR)

    _tokenizer_mental = AutoTokenizer.from_pretrained(_MENTALBERT_DIR)
    _model_mental = AutoModelForSequenceClassification.from_pretrained(_MENTALBERT_DIR)

    _device = torch.device("cpu")
    _model_distil.to(_device)
    _model_mental.to(_device)
    _model_distil.eval()
    _model_mental.eval()

    def _transformer_predict(text, tokenizer, model):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_idx = int(torch.argmax(logits, dim=-1).cpu().numpy()[0])
        return _to_class_name(pred_idx)

    def predict_distilbert(text):
        return _transformer_predict(text, _tokenizer_distil, _model_distil)

    def predict_mentalbert(text):
        return _transformer_predict(text, _tokenizer_mental, _model_mental)

except Exception as e:
    # Keep app working even if transformer deps/models are missing
    def predict_distilbert(text):
        return f"DistilBERT_error: {e}"

    def predict_mentalbert(text):
        return f"MentalBERT_error: {e}"


# -----------------------------
# CNN + LSTM (Keras .h5)
# Assumes models expect numeric input shape (batch, 100)
# -----------------------------
try:
    import numpy as np
    import tensorflow as tf

    _CNN_PATH = "models/cnn_model.h5"
    _LSTM_PATH = "models/LSTM_model.h5"

    _cnn_model = tf.keras.models.load_model(_CNN_PATH)
    _lstm_model = tf.keras.models.load_model(_LSTM_PATH)

    def _text_to_fixed_vector(text, max_len=100):
        """Fallback preprocessing to make CNN/LSTM run.
        It converts text to a simple sequence of word-char hashes mapped into [0,1].
        NOTE: For best accuracy you must use the same tokenizer/embedding used in training.
        """
        text = str(text)
        if not text:
            return np.zeros((max_len,), dtype=np.float32)

        tokens = text.split()[:max_len]
        vec = np.zeros((max_len,), dtype=np.float32)
        for i, tok in enumerate(tokens):
            vec[i] = (abs(hash(tok)) % 10000) / 10000.0
        return vec

    def _keras_predict(model, text):
        x = _text_to_fixed_vector(text)
        x = np.expand_dims(x, axis=0)  # (1, 100)
        preds = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        return _to_class_name(pred_idx)

    def predict_cnn(text):
        return _keras_predict(_cnn_model, text)

    def predict_lstm(text):
        return _keras_predict(_lstm_model, text)

except Exception as e:
    def predict_cnn(text):
        return f"CNN_error: {e}"

    def predict_lstm(text):
        return f"LSTM_error: {e}"





# import joblib
# import numpy as np
# import pickle

# # -----------------------------
# # Load TF-IDF + ML models
# # -----------------------------
# tfidf = joblib.load("vectorizer/tfidf_vectorizer.pkl")

# svm_model = joblib.load("models/svm_stress_classifier_final.pkl")
# lr_model  = joblib.load("models/logistic_regression.pkl")

# # -----------------------------
# # Unified label mapping
# # (MUST match training encoding)
# # -----------------------------
# LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}

# def decode_label(pred):
#     return LABEL_MAP[int(pred)]


# # -----------------------------
# # SVM + Logistic Regression
# # -----------------------------
# def predict_svm(text):
#     X = tfidf.transform([text])
#     pred = svm_model.predict(X)[0]
#     return decode_label(pred)


# def predict_lr(text):
#     X = tfidf.transform([text])
#     pred = lr_model.predict(X)[0]
#     return decode_label(pred)


# # -----------------------------
# # Transformer models (DistilBERT + MentalBERT)
# # -----------------------------
# try:
#     import torch
#     from transformers import AutoTokenizer, AutoModelForSequenceClassification

#     DISTILBERT_PATH = "models/distilbert_stress_model"
#     MENTALBERT_PATH = "models/mentalbert-finetuned"

#     tokenizer_distil = AutoTokenizer.from_pretrained(DISTILBERT_PATH)
#     model_distil     = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_PATH)

#     tokenizer_mental = AutoTokenizer.from_pretrained(MENTALBERT_PATH)
#     model_mental     = AutoModelForSequenceClassification.from_pretrained(MENTALBERT_PATH)

#     device = torch.device("cpu")
#     model_distil.to(device).eval()
#     model_mental.to(device).eval()

#     def _transformer_predict(text, tokenizer, model):
#         inputs = tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             padding=True
#         )
#         inputs = {k: v.to(device) for k, v in inputs.items()}

#         with torch.no_grad():
#             outputs = model(**inputs)
#             pred    = torch.argmax(outputs.logits, dim=-1).item()

#         return decode_label(pred)

#     def predict_distilbert(text):
#         return _transformer_predict(text, tokenizer_distil, model_distil)

#     def predict_mentalbert(text):
#         return _transformer_predict(text, tokenizer_mental, model_mental)

# except Exception as e:
#     def predict_distilbert(text):
#         return f"DistilBERT_NOT_LOADED: {str(e)}"

#     def predict_mentalbert(text):
#         return f"MentalBERT_NOT_LOADED: {str(e)}"


# # -----------------------------
# # CNN + LSTM  ✅ FIXED
# # Loads the SAME tokenizer used during training
# # -----------------------------
# try:
#     import tensorflow as tf
#     from tensorflow.keras.preprocessing.sequence import pad_sequences

#     CNN_PATH  = "models/cnn_model.h5"
#     LSTM_PATH = "models/lstm_model.h5"

#     # ✅ Load the tokenizer saved during training — NOT a new one
#     KERAS_TOKENIZER_PATH = "vectorizer/keras_tokenizer.pkl"
#     with open(KERAS_TOKENIZER_PATH, "rb") as f:
#         keras_tokenizer = pickle.load(f)

#     # ✅ These must match the values used during training exactly
#     MAX_LEN   = 100   # change if your training used a different maxlen
#     NUM_WORDS = 5000  # change if your training used a different vocab size

#     cnn_model  = tf.keras.models.load_model(CNN_PATH)
#     lstm_model = tf.keras.models.load_model(LSTM_PATH)

#     def _keras_predict(model, text):
#         # Tokenize using the SAME tokenizer from training
#         seq    = keras_tokenizer.texts_to_sequences([text])
#         padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
#         pred   = model.predict(padded, verbose=0)
#         return decode_label(np.argmax(pred))

#     def predict_cnn(text):
#         return _keras_predict(cnn_model, text)

#     def predict_lstm(text):
#         return _keras_predict(lstm_model, text)

# except Exception as e:
#     _keras_error = str(e)  # capture BEFORE it goes out of scope
#     print(f"[ERROR] CNN/LSTM failed to load: {_keras_error}")  # shows in terminal

#     def predict_cnn(text):
#         return f"CNN_NOT_LOADED: {_keras_error}"

#     def predict_lstm(text):
#         return f"LSTM_NOT_LOADED: {_keras_error}"


# import joblib
# import numpy as np

# # -----------------------------
# # Load TF-IDF + ML models
# # -----------------------------
# tfidf = joblib.load("vectorizer/tfidf.pkl")

# svm_model = joblib.load("models/svm_stress_classifier_final.pkl")
# lr_model = joblib.load("models/logistic_regression.pkl")

# # -----------------------------
# # Unified label mapping
# # (MUST match training encoding)
# # -----------------------------
# LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}

# def decode_label(pred):
#     return LABEL_MAP[int(pred)]


# # -----------------------------
# # SVM + Logistic Regression
# # -----------------------------
# def predict_svm(text):
#     X = tfidf.transform([text])
#     pred = svm_model.predict(X)[0]
#     return decode_label(pred)


# def predict_lr(text):
#     X = tfidf.transform([text])
#     pred = lr_model.predict(X)[0]
#     return decode_label(pred)


# # -----------------------------
# # Transformer models (DistilBERT + MentalBERT)
# # -----------------------------
# try:
#     import torch
#     from transformers import AutoTokenizer, AutoModelForSequenceClassification

#     DISTILBERT_PATH = "models/distilbert_stress_model"
#     MENTALBERT_PATH = "models/mentalbert-finetuned"

#     tokenizer_distil = AutoTokenizer.from_pretrained(DISTILBERT_PATH)
#     model_distil = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_PATH)

#     tokenizer_mental = AutoTokenizer.from_pretrained(MENTALBERT_PATH)
#     model_mental = AutoModelForSequenceClassification.from_pretrained(MENTALBERT_PATH)

#     device = torch.device("cpu")

#     model_distil.to(device).eval()
#     model_mental.to(device).eval()


#     def _transformer_predict(text, tokenizer, model):
#         inputs = tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             padding=True
#         )

#         inputs = {k: v.to(device) for k, v in inputs.items()}

#         with torch.no_grad():
#             outputs = model(**inputs)
#             logits = outputs.logits
#             pred = torch.argmax(logits, dim=-1).item()

#         return decode_label(pred)


#     def predict_distilbert(text):
#         return _transformer_predict(text, tokenizer_distil, model_distil)


#     def predict_mentalbert(text):
#         return _transformer_predict(text, tokenizer_mental, model_mental)

# except Exception as e:

#     def predict_distilbert(text):
#         return f"DistilBERT_NOT_LOADED: {str(e)}"


#     def predict_mentalbert(text):
#         return f"MentalBERT_NOT_LOADED: {str(e)}"


# # -----------------------------
# # CNN + LSTM (FIXED VERSION)
# # IMPORTANT: must use SAME preprocessing as training
# # -----------------------------
# try:
#     import tensorflow as tf

#     CNN_PATH = "models/cnn_model.h5"
#     LSTM_PATH = "models/lstm_model.h5"

#     cnn_model = tf.keras.models.load_model(CNN_PATH)
#     lstm_model = tf.keras.models.load_model(LSTM_PATH)

#     def _keras_predict(model, text):
#         """
#         IMPORTANT:
#         Replace this with your REAL tokenizer from training (Tokenizer / pad_sequences).
#         This is ONLY a placeholder if no tokenizer was saved.
#         """
#         from tensorflow.keras.preprocessing.sequence import pad_sequences
#         from tensorflow.keras.preprocessing.text import Tokenizer

#         tokenizer = Tokenizer(num_words=5000)
#         tokenizer.fit_on_texts([text])

#         seq = tokenizer.texts_to_sequences([text])
#         padded = pad_sequences(seq, maxlen=100)

#         pred = model.predict(padded, verbose=0)
#         return decode_label(np.argmax(pred))


#     def predict_cnn(text):
#         return _keras_predict(cnn_model, text)


#     def predict_lstm(text):
#         return _keras_predict(lstm_model, text)

# except Exception as e:

#     def predict_cnn(text):
#         return f"CNN_NOT_LOADED: {str(e)}"


#     def predict_lstm(text):
#         return f"LSTM_NOT_LOADED: {str(e)}"


