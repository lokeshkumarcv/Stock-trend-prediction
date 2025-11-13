**Fine-Grained Sentiment Analysis  Stock Market Recommendation Using Enhanced Transformer**



This project predicts whether the stock market will go Up 📈, Down 📉, or remain Stable 🔁, based on financial news headlines. It combines Natural Language Processing (NLP) with machine learning models like LSTM and RoBERTa, and displays predictions through a web interface built with Flask, HTML, and CSS.

**🧠 Features**
 Accepts user input (financial news)

 Predicts market movement using RoBERTa sentiment analysis

 Preprocesses text using SpaCy, NLTK, and custom logic

 Visualizes label distribution and model accuracy (via pie chart and confusion matrix)

 User-friendly web interface using HTML + CSS (Netflix-style login)

**🧩 Technologies Used**
Area	            Tools & Libraries
Backend	            Python, Flask
Machine Learning	TensorFlow/Keras (LSTM), Hugging Face Transformers (RoBERTa)
NLP	                SpaCy, NLTK, contractions
Frontend	        HTML, CSS
Visualization	    Matplotlib
Data            	CSV format (Top1–Top25 news headlines + Label)

**📂 Project Structure**

/static
  └── Stock-News-Dataset/Dataset.csv
  └── Output/ConfusionMatrix.png
  └── Output/StockLabels.png
/templates
  └── Login.html
  └── Dashboard.html
  └── Prediction.html
  └── DatasetInfo.html

**✅ How It Works**
Reads news headlines from Dataset.csv

Preprocesses the text (stopword removal, lemmatization, etc.)

Trains an LSTM model or uses pretrained RoBERTa

Predicts sentiment → maps it to stock movement

Displays prediction and charts on the web app




**👩‍💻 Author**
Boddu Sravanthi
Team Lead – Full Stack + AI Integration


**📃 License**
This project is for academic and learning purposes.
