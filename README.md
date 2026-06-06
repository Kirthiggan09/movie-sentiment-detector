# Movie Review Sentiment Detector 🎬

This is an intelligent text analysis application that automatically classifies movie reviews into positive or negative sentiments. It was built as part of the **SAIA 2163 Final Project (Theme 3)**.

## 🚀 Features
- **Sentiment Prediction**: Enter any movie review and get an instant sentiment prediction (Positive/Negative) along with a confidence score.
- **Explainable AI**: See which specific words in your review influenced the prediction the most.
- **Data Explorer**: View dataset statistics and explore the IMDB 50k dataset.
- **Visualizations**: Interactive charts including word clouds, data distributions, and model performance comparisons.
- **Model Info**: Detailed breakdown of NLP pipeline and model evaluation metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).

## 🛠️ Tech Stack
- **Web App**: Streamlit
- **Machine Learning**: Scikit-learn, Gensim (Word2Vec)
- **NLP**: NLTK
- **Visualizations**: Plotly, Matplotlib, Seaborn, WordCloud
- **Data Processing**: Pandas, NumPy, HuggingFace Datasets

## 📂 Project Structure
```text
movie-sentiment-detector/
├── app.py                  # Main Streamlit application
├── train_models.py         # Script to download data, train, and save models
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore file
├── data/                   # Directory containing dataset (created during training)
└── models/                 # Directory containing saved models and results (created during training)
```

## ⚙️ Setup and Installation

1. **Clone the repository (or navigate to the folder)**
   ```bash
   cd movie-sentiment-detector
   ```

2. **Install dependencies**
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Models and Download Data**
   Before running the app, you need to download the IMDB dataset and train the models. Run the training script:
   ```bash
   python train_models.py
   ```
   *Note: This process may take several minutes depending on your machine.*

4. **Run the Streamlit App**
   Once training is complete, start the web app:
   ```bash
   streamlit run app.py
   ```
   The app will open automatically in your browser at `http://localhost:8501`.

## 🧠 NLP Pipeline Overview
1. **Preprocessing**: Lowercasing, HTML tag removal, URL removal, special character removal.
2. **Tokenization**: Splitting text into individual words.
3. **Stopword Removal**: Removing non-informative words (e.g., 'the', 'is', 'in').
4. **Lemmatization**: Reducing words to their base form.
5. **Feature Extraction**: Implemented both **TF-IDF** and **Word2Vec**.
6. **Models Trained**: Naive Bayes, Logistic Regression, SVM, Random Forest.
