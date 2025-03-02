import streamlit as st
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import numpy as np

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the BERT model (use sigmoid activation for multi-label classification)
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10, problem_type="multi_label_classification")

# Load the model weights
try:
    model.load_weights('./models/bert-weights.weights.h5', by_name=True)
    st.write("Model loaded successfully!")
except ValueError as e:
    st.error(f"Error loading weights: {e}")

# Define emotions list
emotions = ['praise', 'amusement', 'anger', 'disapproval', 'confusion', 'interest', 'sadness', 'fear', 'joy', 'love']

def preprocess_input(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=512)
    return inputs

def predict_emotions(text, threshold=0.6, top_n=3):
    inputs = preprocess_input(text)
    predictions = model(inputs)
    
    # Apply sigmoid activation and get probabilities
    prob = tf.sigmoid(predictions.logits)
    
    # Convert to numpy array for easier manipulation
    prob = prob.numpy().flatten()
    
    # Get indices where the probability exceeds the threshold
    filtered_indices = np.where(prob >= threshold)[0]
    
    # If no emotion exceeds the threshold, return top N emotions based on probability
    if len(filtered_indices) == 0:
        # Sort emotions by probability and select the top N
        sorted_indices = np.argsort(prob)[::-1][:top_n]
        predicted_emotions = [emotions[i] for i in sorted_indices]
    else:
        # Otherwise, return the emotions with probability above the threshold
        predicted_emotions = [emotions[i] for i in filtered_indices]
    
    return predicted_emotions

# Streamlit UI
st.title("Emotion Detection from Text")
st.write("Enter text to predict the emotions:")

# Text input box
user_input = st.text_area("Text Input", height=100)

# Predict button
if st.button("Predict Emotions"):
    if user_input:
        predicted_emotions = predict_emotions(user_input)
        if predicted_emotions:
            st.write(f"The predicted emotions are: {', '.join(predicted_emotions)}")
        else:
            st.write("No relevant emotions detected.")
    else:
        st.write("Please enter some text to predict emotions.")
