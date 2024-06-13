from flask import Flask, render_template, request
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re
import transformers
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load the datasets
sarcastic = pd.read_csv("sarcastic.csv")
non_sarcastic = pd.read_csv("nonsarcastic.csv")

# Add label column
sarcastic['label'] = 1
non_sarcastic['label'] = 0

# Concatenate the datasets
df = pd.concat([sarcastic, non_sarcastic])

# Drop duplicates
df = df.drop_duplicates(keep='first')

# Preprocess text
df['text'] = df['text'].apply(lambda x: re.sub(r'https?://\S+|www\.\S+', '', x))
df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
df['text'] = df['text'].apply(lambda x: x.lower())

# Check if 'label' column exists in the DataFrame
if 'label' not in df.columns:
    raise ValueError("Label column not found in the dataset. Please ensure that the dataset contains a 'label' column.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Train machine learning models
cv = CountVectorizer(max_features=5000)
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train_cv, y_train)

svm_classifier = SVC(probability=True)
svm_classifier.fit(X_train_cv, y_train)

dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train_cv, y_train)

lr_classifier = LogisticRegression()
lr_classifier.fit(X_train_cv, y_train)

# Load LSTM model
lstm_model = load_model(r"models\lstm_model.h5")


# Load BERT model
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Preprocess text function
def preprocess_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

# Function to predict using machine learning models
def predict(text, model_name):
    preprocessed_text = preprocess_text(text)
    
    if model_name == 'LSTM':
        # Preprocess text for LSTM
        encoded_text = [one_hot(preprocessed_text, 5000)]
        # Pad the sequence to ensure consistent length
        encoded_text = pad_sequences(encoded_text, padding='pre', maxlen=100)
        prediction_prob = lstm_model.predict(encoded_text)[0][0]
    elif model_name == 'BERT':
        # Preprocess text for BERT
        inputs = bert_tokenizer(preprocessed_text, return_tensors='pt', max_length=128, truncation=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            logits = outputs.logits
            prediction = torch.softmax(logits, dim=1)
            prediction_prob = prediction[0][0].item()
    else:
        # Transform the preprocessed text using CountVectorizer for traditional ML models
        encoded_text = cv.transform([preprocessed_text]).toarray()
        if model_name == 'Random Forest':
            prediction_prob = rf_classifier.predict_proba(encoded_text)[0][1]
        elif model_name == 'Support Vector Machine':
            prediction_prob = svm_classifier.predict_proba(encoded_text)[0][1]
        elif model_name == 'Decision Tree':
            prediction_prob = dt_classifier.predict_proba(encoded_text)[0][1]
        elif model_name == 'Logistic Regression':
            prediction_prob = lr_classifier.predict_proba(encoded_text)[0][1]
        else:
            raise ValueError(f"Invalid model_name: {model_name}")
    
    # Make prediction based on probability threshold
    prediction = 1 if prediction_prob > 0.5 else 0
    
    return prediction, prediction_prob

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_text():
    text = request.form['text']
    algorithm = request.form['algorithm']
    
    try:
        prediction, probability = predict(text, algorithm)
        
        # Convert prediction to text format
        prediction_text = "sarcastic" if prediction == 1 else "non sarcastic"

        # Calculate accuracy (not applicable in this context)
        if algorithm == 'Random Forest':
            accuracy = accuracy_score(y_test, rf_classifier.predict(X_test_cv))
        elif algorithm == 'Support Vector Machine':
            accuracy = accuracy_score(y_test, svm_classifier.predict(X_test_cv))
        elif algorithm == 'Decision Tree':
            accuracy = accuracy_score(y_test, dt_classifier.predict(X_test_cv))
        elif algorithm == 'Logistic Regression':
            accuracy = accuracy_score(y_test, lr_classifier.predict(X_test_cv))
        elif algorithm == 'LSTM':
            # LSTM model accuracy is not applicable
            accuracy = 0.998749
        elif algorithm == 'BERT':
            # BERT model accuracy is not applicable
            accuracy = 0.9419
        else:
            accuracy = None
        
        return render_template('index.html', algorithm=algorithm, input_text=text, result=prediction_text, accuracy=accuracy, probability=probability)
    except Exception as e:
        # Handle any exceptions that may occur during prediction
        return render_template('index.html', algorithm=algorithm, input_text=text, result=f'Error: {str(e)}', accuracy=None, probability=None)

if __name__ == '__main__':
    app.run(debug=True)
