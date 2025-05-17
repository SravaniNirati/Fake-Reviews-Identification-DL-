import streamlit as st
import pickle
import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load LSTM, CNN, and RNN models and tokenizers with error handling
try:
    lstm_model = tf.keras.models.load_model("lstm_fake_review_model.keras")
    with open("tokenizer_lstm.pkl", "rb") as f:
        lstm_tokenizer = pickle.load(f)
    
    cnn_model = tf.keras.models.load_model("cnn_fake_review_model.keras")
    with open("tokenizer_cnn.pkl", "rb") as f:
        cnn_tokenizer = pickle.load(f)
    
    # Load the RNN model (PyTorch) and tokenizer
    with open("tokenizer_rnn.pkl", "rb") as f:
        rnn_tokenizer = pickle.load(f)
    
    # Recreate the RNN model and load state dict
    class SimpleRNN(torch.nn.Module):
        def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
            super(SimpleRNN, self).__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embed_size)
            self.rnn = torch.nn.RNN(embed_size, hidden_size, batch_first=True)
            self.fc = torch.nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            x = self.embedding(x)
            _, h_n = self.rnn(x)
            return self.fc(h_n.squeeze(0))
    
    # Define parameters for the RNN model
    vocab_size = rnn_tokenizer.vocab_size
    embed_size = 64
    hidden_size = 32
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    rnn_model = SimpleRNN(vocab_size, embed_size, hidden_size, num_classes)
    rnn_model.load_state_dict(torch.load("rnn_fake_review_model.pth", map_location=device))
    rnn_model.to(device)
    rnn_model.eval()

    st.write("LSTM, CNN, RNN and BERT models loaded successfully!")
except Exception as e:
    st.error(f"Error loading LSTM, CNN, or RNN models or tokenizers: {e}")

# Load the BERT model and tokenizer
@st.cache_resource
def load_bert_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("SravaniNirati/bert_fake_review_detection")
        model = AutoModelForSequenceClassification.from_pretrained("SravaniNirati/bert_fake_review_detection")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading BERT model: {e}")
        return None, None

bert_tokenizer, bert_model = load_bert_model()

# Function to predict using the LSTM or CNN model (Keras)
def predict_review_keras(review_text, model, tokenizer, maxlen=100):
    sequences = tokenizer.texts_to_sequences([review_text])
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    prediction = model.predict(padded_sequences)
    return "Real Review" if prediction[0][0] >= 0.5 else "Fake Review"

# Function to predict using the RNN model (PyTorch)
def predict_review_rnn(review_text, model, tokenizer, maxlen=128):
    inputs = tokenizer([review_text], truncation=True, padding=True, max_length=maxlen, return_tensors='pt')['input_ids']
    inputs = inputs.to(device)
    outputs = model(inputs)
    _, prediction = torch.max(outputs, 1)
    return "Real Review" if prediction.item() == 1 else "Fake Review"

# Function to classify a review using the BERT model
def classify_review_bert(review_text, tokenizer, model):
    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities).item()
    
    labels = {0: "❌ Fake Review", 1: "✅ Real Review"}
    
    return labels[predicted_class], probabilities.tolist()

# Create tabs for the models
tab1, tab2, tab3, tab4 = st.tabs(["LSTM Model", "CNN Model", "RNN Model", "BERT Model"])

# Tab for the LSTM Model
with tab1:
    st.header("LSTM Fake Review Detection")
    lstm_review = st.text_area("Enter a review text for LSTM Model:")
    
    if st.button("Predict with LSTM Model"):
        if lstm_review:
            try:
                result = predict_review_keras(lstm_review, lstm_model, lstm_tokenizer)
                st.write(f"Prediction: {result}")
            except Exception as e:
                st.error(f"Error during LSTM prediction: {e}")
        else:
            st.warning("Please enter a review text.")

# Tab for the CNN Model
with tab2:
    st.header("CNN Fake Review Detection")
    cnn_review = st.text_area("Enter a review text for CNN Model:")
    
    if st.button("Predict with CNN Model"):
        if cnn_review:
            try:
                result = predict_review_keras(cnn_review, cnn_model, cnn_tokenizer)
                st.write(f"Prediction: {result}")
            except Exception as e:
                st.error(f"Error during CNN prediction: {e}")
        else:
            st.warning("Please enter a review text.")

# Tab for the RNN Model
with tab3:
    st.header("RNN Fake Review Detection")
    rnn_review = st.text_area("Enter a review text for RNN Model:")
    
    if st.button("Predict with RNN Model"):
        if rnn_review:
            try:
                result = predict_review_rnn(rnn_review, rnn_model, rnn_tokenizer)
                st.write(f"Prediction: {result}")
            except Exception as e:
                st.error(f"Error during RNN prediction: {e}")
        else:
            st.warning("Please enter a review text.")

# Tab for the BERT Model
with tab4:
    st.header("BERT Fake Review Detection")
    bert_review = st.text_area("Enter a review text for BERT Model:")
    
    if st.button("Predict with BERT Model"):
        if bert_review:
            try:
                result, probs = classify_review_bert(bert_review, bert_tokenizer, bert_model)
                st.write(f"Prediction: {result}")
                st.write(f"Confidence Scores: {probs}")
            except Exception as e:
                st.error(f"Error during BERT prediction: {e}")
        else:
            st.warning("Please enter a review text.")

