Fake Reviews Identification using Deep Learning Techniques
This project focuses on detecting fake product reviews using multiple deep learning models: CNN, RNN, LSTM, and BERT. A Streamlit web application is developed to demonstrate real-time predictions using all four models.

Problem Statement:
Online product reviews significantly influence customer decisions. However, the presence of fake reviews misleads consumers and harms businesses. This project aims to identify and classify reviews as Fake or Real using deep learning.

Technologies Used:
Python

TensorFlow (for CNN, LSTM models)

PyTorch (for RNN model)

Hugging Face Transformers (for BERT)

Streamlit (for interactive web interface)

Pickle (for saving tokenizers)

Keras Preprocessing (for LSTM/CNN input preparation)

Models Implemented"
🔸 Convolutional Neural Network (CNN)
Text classification using convolutional filters.

Accuracy: 94.11%

🔸 Recurrent Neural Network (RNN)
Vanilla RNN with embedding and linear layer.

Accuracy: 57.94%

🔸 Long Short-Term Memory (LSTM)
Captures long-term dependencies in text.

Accuracy: 94.21%

🔸 BERT (Bidirectional Encoder Representations from Transformers)
Pretrained transformer model from Hugging Face.BERT significantly outperforms traditional RNN-based architectures, demonstrating its capability in understanding contextual and semantic nuances of review texts. Fine-tuned on fake review dataset.

Accuracy: 98.00%

💻 Streamlit Web App
An interactive web app lets users input custom review texts and classify them using all four models.

App Features
✅ Tabs for each model

✅ Real-time predictions

✅ Displays prediction and confidence

✅ Error handling

The app loads models from the following files:

lstm_fake_review_model.keras

cnn_fake_review_model.keras

rnn_fake_review_model.pth

BERT model: SravaniNirati/bert_fake_review_detection from Hugging Face

Tokenizers: tokenizer_lstm.pkl, tokenizer_cnn.pkl, tokenizer_rnn.pkl

Future Work:
Improve RNN performance with GRU or bi-directional layers

Integrate Explainable AI (LIME/SHAP) for model interpretation

Add ensemble model for better accuracy

👨‍💻Author
Sravani Nirati
