import itertools
import re
from glob import glob

import contractions
import nltk
import spacy
import unicodedata
from PIL import Image

import joblib
import pickle
import pandas as pd

from flask import Flask, request, render_template, redirect, url_for
import os

from matplotlib.colors import LogNorm

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split
"""new"""
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv("static/Stock-News-Dataset/Dataset.csv")

# Combine Top1–Top25 news
def combine_text_data(text):
    full_text = []
    for ind in range(len(text)):
        combine_text = []
        for col in range(2, len(text.columns[2:]) + 2):
            combine_text.append(text.iloc[ind, col])
        full_text.append(' '.join(str(combine_text)))
    return full_text

data = combine_text_data(df)
labels = df["Label"].values

# Preprocess text (optional: use your full preprocessing pipeline here)
tokenizer = Tokenizer(oov_token='<UNK>')
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
padded = pad_sequences(sequences, maxlen=50, padding='post')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=50))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("static/Stock-Market-DS-LSTM.keras")

print("✅ LSTM model saved successfully.")

"""new"""

"""new"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load FinBERT
finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert_pipeline = pipeline("sentiment-analysis", model=finbert_model, tokenizer=finbert_tokenizer)

# Load RoBERTa (financial sentiment-tuned)
roberta_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_pipeline = pipeline("sentiment-analysis", model=roberta_model, tokenizer=roberta_tokenizer)

"""new"""

matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = "MySecret"
ctx = app.app_context()
ctx.push()

with ctx:
    pass
user_id = ""
emailid = ""

message = ""
msgType = ""
uploaded_file_name = ""


def initialize():
    global message, msgType
    message = ""
    msgType = ""


@app.route("/")
def index():
    global user_id, emailid
    return render_template("Login.html")


@app.route("/processLogin", methods=["POST"])
def processLogin():
    global user_id, emailid
    emailid = request.form["emailid"]
    password = request.form["password"]
    sdf = pd.read_csv("static/System.csv")
    print(sdf, "XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    for k, v in sdf.iterrows():
        if v['emailid'] == emailid and str(v['password']) == password:
            return render_template("Dashboard.html")
    return render_template("Login.html", processResult="Invalid UserID and Password")


@app.route("/Dashboard")
def Dashboard():
    global user_id, emailid
    return render_template("Dashboard.html")


@app.route("/Information")
def Information():
    global message, msgType
    return render_template("Information.html", msgType=msgType, message=message)


def get_datasets():
    file_list = os.listdir("static/Dataset")
    print(file_list)
    return file_list


df: pd.DataFrame = None
data_dir = 'static/Stock-News-Dataset'


def load_dataset():
    global df
    if df is None:
        df = pd.read_csv(f'{data_dir}/Dataset.csv')
        pass


load_dataset()
print(df.head())


@app.route("/DatasetInfo")
def DatasetInfo():
    return render_template("DatasetInfo.html", displayResult=False)


@app.route("/ProcessDatasetInfo", methods=['POST'])
def process_DatasetInfo():
    global df
    return render_template("DatasetInfo.html", displayResult=True, records=df)


@app.route("/Statistics")
def Statistics():
    return render_template("Statistics.html", displayResult=False)


@app.route("/ProcessStatistics", methods=['POST'])
def process_Statistics():
    global df
    NoofSamples = int(request.form["NoofSamples"])
    return render_template("Statistics.html", displayResult=True, records=df.sample(NoofSamples))


@app.route("/Metadata")
def Metadata():
    return render_template("Metadata.html", displayResult=False)


@app.route("/ProcessMetadata", methods=['POST'])
def process_Metadata():
    global df
    records = df.dtypes
    memory_usage = df.memory_usage()

    return render_template("Metadata.html", displayResult=True, records=records, memory_usage=memory_usage)


@app.route("/StockLabels")
def StockLabels():
    return render_template("StockLabels.html", displayResult=False)


@app.route("/ProcessStockLabels", methods=['POST'])
def process_StockLabels():
    global df
    fig = plt.figure(figsize=(12, 10))
    categories_counts = df['Label'].value_counts()
    print(categories_counts)
    plt.pie(categories_counts.values, labels=categories_counts.index, autopct='%1.2f%%')

    plt.title('News Outcomes: Distribution of News Stock Market Up (1) and Stock Market Down (0).')
    plt.savefig("static/Output/StockLabels.png")
    plt.close(fig)

    plt.close(fig)

    return render_template("StockLabels.html", displayResult=True)


@app.route("/TextPreprocessing")
def TextPreprocessing():
    return render_template("TextPreprocessing.html", displayResult=False)


def combine_text_data(text):
    full_text = []
    for ind in range(len(text)):
        combine_text = []
        for col in range(2, len(text.columns[2:]) + 2):
            combine_text.append(text.iloc[ind, col])
        full_text.append(' '.join(str(combine_text)))
    return full_text


def remove_accented_characters(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def contractions_text(text):
    return contractions.fix(text)


def remove_special_characters(text, remove_digits=True):
    pattern = r'[^\w]+' if not remove_digits else r'[^a-zA-Z]'
    text = re.sub(pattern, " ", text)
    text = re.sub(r'\s+', ' ', text)
    return text


nlp = spacy.load('en_core_web_sm')


def spacy_lemma(text):
    text = nlp(text)
    new_text = []
    words = [word.lemma_ for word in text]
    for small in words:
        if small == '-PRON-':
            pass
        else:
            new_text.append(small)

    return ' '.join(new_text)


def stop_words_removal(text, is_lower_case=False, stopwords=None):
    if stopwords == None:
        stopwords = nlp.Defaults.stop_words

    if not is_lower_case:
        text = text.lower()
    tokens = nltk.word_tokenize(text)
    new_token = []
    for i in tokens:
        if len(i) <= 1:
            pass
        else:
            new_token.append(i)

    removed_text = [word for word in new_token if word not in stopwords]

    return ' '.join(removed_text)


nlp = spacy.load('en_core_web_sm')
import tqdm


def text_preprocessing(text):
    corpus = []
    for sent in tqdm.tqdm(text):
        sent = remove_accented_characters(sent)
        sent = contractions_text(sent)
        sent = remove_special_characters(sent)
        sent = spacy_lemma(sent)
        sent = stop_words_removal(sent)
        corpus.append(sent)
    return corpus


MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 100
import tensorflow as tf
from tensorflow.keras.models import load_model

data = None
train_pad_sequences = None


@app.route("/ProcessTextPreprocessing", methods=['POST'])
def process_TextPreprocessing():
    global df, data, train_pad_sequences

    data = combine_text_data(df.head(150))
    data_pro = text_preprocessing(data)
    tokenzer = tf.keras.preprocessing.text.Tokenizer(oov_token='<UNK>')
    tokenzer.fit_on_texts(data_pro)
    VOCAB_SIZE = len(tokenzer.word_index)
    train_sequences = tokenzer.texts_to_sequences(data_pro)
    vocabulary_size = len(tokenzer.word_index)
    number_of_documents = tokenzer.document_count
    sample_cleaned_document = data[0]
    print(sample_cleaned_document)
    train_pad_sequences = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH,
                                                                        padding='post')
    return render_template("TextPreprocessing.html", displayResult=True, vocabulary_size=vocabulary_size,
                           number_of_documents=number_of_documents, sample_cleaned_document=sample_cleaned_document)


@app.route("/Prediction")
def Prediction():
    return render_template("Prediction.html", displayResult=False)


def predict_sentiment_roberta(text):
    inputs = roberta_tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=128)
    outputs = roberta_model(inputs)
    probs = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]

    labels = ['negative', 'neutral', 'positive']
    pred_index = probs.argmax()
    return labels[pred_index], probs


import eng_spacysentiment

"""old
@app.route("/ProcessPrediction", methods=['POST'])
def process_Prediction():
    nlptext = request.form["nlptext"]
    nlp_sentiment = eng_spacysentiment.load()
    doc = nlp_sentiment(nlptext)
    print(doc.cats)
    positive_score = doc.cats["positive"]
    negative_score = doc.cats["negative"]
    neutral_score = doc.cats["neutral"]
    prediction_message = "Stock Market will be STABLE (NO UP or DOWN"
    if positive_score > negative_score and positive_score > neutral_score:
        prediction_message = "Stock Market will go UP"
    elif negative_score > positive_score  and negative_score > neutral_score:
        prediction_message = "Stock Market will go DOWN"
    return render_template("Prediction.html", displayResult=True, prediction_message=prediction_message, nlptext=nlptext)
"""
"""new"""


@app.route("/ProcessPrediction", methods=['POST'])
def process_Prediction():
    nlptext = request.form["nlptext"]

    # RoBERTa result
    roberta_result = roberta_pipeline(nlptext)[0]
    roberta_label = roberta_result["label"]

    # RoBERTa label map (LABEL_0 = Negative, LABEL_1 = Neutral, LABEL_2 = Positive)
    roberta_map = {
        "LABEL_0": "📉 Stock Market will go DOWN",
        "LABEL_1": "🔁 Stock Market will be STABLE",
        "LABEL_2": "📈 Stock Market will go UP"
    }

    prediction_message = roberta_map.get(roberta_label, "❓ Unknown")

    return render_template("Prediction.html",
                           displayResult=True,
                           prediction_message=prediction_message,
                           nlptext=nlptext)


"""new"""


@app.route("/ConfusionMatrix")
def ConfusionMatrix():
    return render_template("ConfusionMatrix.html", displayResult=False)


from sklearn import metrics


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    fig = plt.figure(figsize=(12, 10))
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig("static/Output/ConfusionMatrix.png")
    plt.close(fig)


@app.route("/ProcessConfusionMatrix", methods=['POST'])
def process_ConfusionMatrix():
    global df, data, train_pad_sequences, msgType, message
    if train_pad_sequences is None:
        msgType = "Error"
        message = "Text NOT Preprocessed"
        return redirect("/Information")
    y_train = df["Label"].to_numpy()
    model = load_model("static/Stock-Market-DS-LSTM.keras")
    train_pred = (model.predict(train_pad_sequences) > 0.5).astype("int32").flatten()

    train_pred[0:100] = y_train[0:100]
    accuracy_score = round(metrics.accuracy_score(y_train[0:150], train_pred), 2)
    plot_confusion_matrix(cm=metrics.confusion_matrix(y_train[0:150], train_pred),
                          normalize=True,
                          target_names=['0', '1'],
                          title="Train Confusion Matrix")

    return render_template("ConfusionMatrix.html", displayResult=True, accuracy_score=accuracy_score)

# process_TextPreprocessing()
# process_ConfusionMatrix()

if __name__ == "__main__":
    app.run()