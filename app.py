import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pickle
import json
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Movie Sentiment Detector", page_icon="🎬", layout="wide")

# Directory paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Ensure NLTK data is downloaded
@st.cache_resource
def download_nltk_data():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except:
            pass

download_nltk_data()

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens), tokens

@st.cache_data
def load_data():
    sample_path = os.path.join(DATA_DIR, 'sample_reviews.csv')
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
    return None

@st.cache_data
def load_stats():
    stats_path = os.path.join(MODEL_DIR, 'dataset_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_eval_results():
    eval_path = os.path.join(MODEL_DIR, 'evaluation_results.json')
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            return json.load(f)
    return None

@st.cache_resource
def load_models():
    models = {}
    try:
        # Load Vectorizer
        with open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
            models['tfidf'] = pickle.load(f)
            
        # Load Best Model (Logistic Regression usually performs well)
        # We'll try to load Logistic Regression as default
        lr_path = os.path.join(MODEL_DIR, 'tfidf_logistic_regression.pkl')
        if os.path.exists(lr_path):
            with open(lr_path, 'rb') as f:
                models['classifier'] = pickle.load(f)
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
    return models

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.title("🎬 Navigation")
page = st.sidebar.radio("Go to", 
    ["Home / About", "Text Analyzer", "Data Explorer", "Visualizations", "Model Info"]
)

# -----------------------------------------------------------------------------
# Page 1: Home / About
# -----------------------------------------------------------------------------
if page == "Home / About":
    st.title("🎬 Movie Review Sentiment Detector")
    st.markdown("---")
    
    st.markdown("""
    ### 📌 Project Overview
    Welcome to the **Movie Review Sentiment Detector**! This intelligent text analysis application automatically classifies movie reviews into positive or negative sentiments.
    
    This project is part of the **SAIA 2163 Final Project (Theme 3)**, demonstrating a complete Natural Language Processing (NLP) system from data processing to an interactive web application.
    
    ### 🎯 What problem does this solve?
    Understanding user sentiment is crucial for businesses. By automatically classifying thousands of movie reviews, platforms can gauge overall audience reaction to a film without manually reading each review.
    
    ### 🚀 How to use this app
    - Navigate to the **Text Analyzer** from the sidebar to test your own movie reviews.
    - Check out the **Data Explorer** to view the IMDB dataset we used.
    - Explore the **Visualizations** to see word clouds and data distributions.
    - Learn about the machine learning models in the **Model Info** section.
    
    ### 👥 Team Members
    *Your Team Name Here*
    - Member 1
    - Member 2
    - Member 3
    - Member 4
    """)
    
    st.info("👈 Use the sidebar to navigate through the application.")

# -----------------------------------------------------------------------------
# Page 2: Text Analyzer
# -----------------------------------------------------------------------------
elif page == "Text Analyzer":
    st.title("🔍 Analyze Your Movie Review")
    st.markdown("Enter a movie review below, and our trained machine learning model will predict whether the sentiment is **Positive** or **Negative**.")
    
    # Load models
    models = load_models()
    
    if not models:
        st.warning("⚠️ Models not found! Please run the training script `train_models.py` first.")
    else:
        user_input = st.text_area("✍️ Type or paste a movie review here:", height=200, 
                                  placeholder="Example: This movie was absolutely fantastic! The acting was brilliant and the plot kept me engaged from start to finish.")
        
        if st.button("Predict Sentiment", type="primary"):
            if user_input.strip() == "":
                st.error("Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing text..."):
                    # Preprocess
                    clean_text, tokens = preprocess_text(user_input)
                    
                    # Vectorize
                    vectorized_text = models['tfidf'].transform([clean_text])
                    
                    # Predict
                    prediction = models['classifier'].predict(vectorized_text)[0]
                    
                    # Get probabilities if available
                    try:
                        probs = models['classifier'].predict_proba(vectorized_text)[0]
                        conf_score = max(probs)
                    except:
                        # Fallback for models like LinearSVC without predict_proba by default
                        d_func = models['classifier'].decision_function(vectorized_text)[0]
                        conf_score = 1 / (1 + np.exp(-d_func)) if prediction == 1 else 1 - (1 / (1 + np.exp(-d_func)))
                    
                    st.markdown("---")
                    st.subheader("📊 Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:
                            st.success(f"### 🟢 POSITIVE SENTIMENT")
                        else:
                            st.error(f"### 🔴 NEGATIVE SENTIMENT")
                            
                    with col2:
                        st.metric("Confidence Score", f"{conf_score:.2%}")
                    
                    # Feature importance (Words that influenced prediction)
                    st.markdown("### 🔑 Key Words Found")
                    feature_names = models['tfidf'].get_feature_names_out()
                    
                    # Get indices of non-zero features in the input text
                    nonzero_indices = vectorized_text.nonzero()[1]
                    
                    if len(nonzero_indices) > 0 and hasattr(models['classifier'], 'coef_'):
                        coefs = models['classifier'].coef_[0]
                        word_impact = []
                        
                        for idx in nonzero_indices:
                            word = feature_names[idx]
                            impact = coefs[idx]
                            word_impact.append({"Word": word, "Impact": impact})
                            
                        impact_df = pd.DataFrame(word_impact)
                        # Sort by absolute impact
                        impact_df['Abs_Impact'] = impact_df['Impact'].abs()
                        impact_df = impact_df.sort_values('Abs_Impact', ascending=False).head(10)
                        
                        # Plot
                        fig = px.bar(impact_df, x='Impact', y='Word', orientation='h',
                                     color='Impact', color_continuous_scale='RdYlGn',
                                     title='Top 10 Influential Words in Your Review')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("Processed Tokens:", ", ".join(tokens[:20]) + ("..." if len(tokens)>20 else ""))

# -----------------------------------------------------------------------------
# Page 3: Data Explorer
# -----------------------------------------------------------------------------
elif page == "Data Explorer":
    st.title("🗂️ Dataset Explorer")
    st.markdown("Explore the IMDB Movie Reviews dataset used to train our models.")
    
    df = load_data()
    stats = load_stats()
    
    if df is None or stats is None:
        st.warning("⚠️ Data not found! Please run the training script `train_models.py` first.")
    else:
        # Key Metrics
        st.subheader("📈 Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews (Full Dataset)", f"{stats['total_reviews']:,}")
        col2.metric("Positive Reviews", f"{stats['positive_reviews']:,}")
        col3.metric("Negative Reviews", f"{stats['negative_reviews']:,}")
        
        st.markdown("---")
        
        # Sample Data
        st.subheader("📝 Sample Data (500 records)")
        st.dataframe(df[['text', 'sentiment', 'clean_text']], use_container_width=True)
        
        # Data Distribution
        st.subheader("📊 Sentiment Class Distribution")
        dist_df = pd.DataFrame({
            'Sentiment': ['Positive', 'Negative'],
            'Count': [stats['positive_reviews'], stats['negative_reviews']]
        })
        fig = px.bar(dist_df, x='Sentiment', y='Count', 
                     color='Sentiment',
                     color_discrete_sequence=['#2ecc71', '#e74c3c'],
                     title='Sentiment Class Distribution — Bar Chart')
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Page 4: Visualizations
# -----------------------------------------------------------------------------
elif page == "Visualizations":
    st.title("📊 Data & Model Visualizations")
    
    stats = load_stats()
    eval_results = load_eval_results()
    
    if stats is None or eval_results is None:
        st.warning("⚠️ Visualization data not found! Please run the training script first.")
    else:
        tab1, tab2, tab3 = st.tabs(["☁️ Word Clouds", "📊 Top Words", "📈 Model Performance"])
        
        with tab1:
            st.subheader("Word Clouds")
            st.markdown("Visualizing the most frequent words in positive vs negative reviews.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Positive Reviews")
                pos_words = stats.get('top_positive_words', {})
                if pos_words:
                    wordcloud_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate_from_frequencies(pos_words)
                    fig_pos, ax_pos = plt.subplots(figsize=(10, 5))
                    ax_pos.imshow(wordcloud_pos, interpolation='bilinear')
                    ax_pos.axis('off')
                    st.pyplot(fig_pos)
            
            with col2:
                st.markdown("#### Negative Reviews")
                neg_words = stats.get('top_negative_words', {})
                if neg_words:
                    wordcloud_neg = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate_from_frequencies(neg_words)
                    fig_neg, ax_neg = plt.subplots(figsize=(10, 5))
                    ax_neg.imshow(wordcloud_neg, interpolation='bilinear')
                    ax_neg.axis('off')
                    st.pyplot(fig_neg)
                    
        with tab2:
            st.subheader("Top 20 Most Frequent Words")
            top_all = stats.get('top_all_words', {})
            if top_all:
                df_top = pd.DataFrame(list(top_all.items())[:20], columns=['Word', 'Frequency'])
                df_top = df_top.sort_values('Frequency', ascending=True)
                
                fig = px.bar(df_top, x='Frequency', y='Word', orientation='h', 
                             title='Top 20 Overall Words (After Preprocessing)',
                             color='Frequency', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
                
        with tab3:
            st.subheader("Model Performance Comparison")
            
            metrics_data = []
            for name, metrics in eval_results.items():
                metrics_data.append({
                    "Model": name.replace('TF-IDF + ', 'TFIDF: ').replace('Word2Vec + ', 'W2V: '),
                    "Accuracy": metrics['accuracy'],
                    "Precision": metrics['precision'],
                    "Recall": metrics['recall'],
                    "F1-Score": metrics['f1_score']
                })
                
            df_metrics = pd.DataFrame(metrics_data)
            df_melted = df_metrics.melt(id_vars='Model', var_name='Metric', value_name='Score')
            
            fig = px.bar(df_melted, x='Model', y='Score', color='Metric', barmode='group',
                         title='Model Performance Comparison (Accuracy, Precision, Recall, F1-Score)',
                         color_discrete_sequence=px.colors.qualitative.Safe)
            fig.update_yaxes(range=[0.7, 1.0])
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Confusion Matrices")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Logistic Regression")
                lr_key = "TF-IDF + Logistic Regression" if "TF-IDF + Logistic Regression" in eval_results else None
                if lr_key:
                    cm_lr = np.array(eval_results[lr_key]['confusion_matrix'])
                    fig_lr = px.imshow(cm_lr, text_auto=True, color_continuous_scale='Blues',
                                       labels=dict(x="Predicted Label", y="True Label"),
                                       x=['Negative', 'Positive'], y=['Negative', 'Positive'])
                    st.plotly_chart(fig_lr, use_container_width=True)
                    
            with col2:
                st.markdown("#### Multinomial Naive Bayes")
                nb_key = "TF-IDF + Naive Bayes" if "TF-IDF + Naive Bayes" in eval_results else None
                if nb_key:
                    cm_nb = np.array(eval_results[nb_key]['confusion_matrix'])
                    fig_nb = px.imshow(cm_nb, text_auto=True, color_continuous_scale='Reds',
                                       labels=dict(x="Predicted Label", y="True Label"),
                                       x=['Negative', 'Positive'], y=['Negative', 'Positive'])
                    st.plotly_chart(fig_nb, use_container_width=True)

# -----------------------------------------------------------------------------
# Page 5: Model Info
# -----------------------------------------------------------------------------
elif page == "Model Info":
    st.title("🧠 Machine Learning Models")
    
    eval_results = load_eval_results()
    
    if eval_results is None:
        st.warning("⚠️ Model evaluation results not found! Please run the training script first.")
    else:
        st.markdown("""
        ### NLP Pipeline
        1. **Text Cleaning**: Lowercasing, removing HTML tags and special characters.
        2. **Tokenization**: Splitting text into individual words.
        3. **Stopword Removal**: Removing common words (the, is, in) that don't add sentiment value.
        4. **Lemmatization**: Converting words to their base form (e.g., 'running' -> 'run').
        5. **Feature Extraction**: Converting text to numbers using **TF-IDF** and **Word2Vec**.
        """)
        
        st.markdown("---")
        st.subheader("📊 Detailed Performance Metrics")
        
        # Create a detailed table of metrics
        metrics_data = []
        for name, metrics in eval_results.items():
            metrics_data.append({
                "Model Configuration": name,
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1-Score": f"{metrics['f1_score']:.4f}"
            })
            
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 Classification Report (Best Model)")
        
        # Get best model based on F1
        best_model = max(eval_results.keys(), key=lambda k: eval_results[k]['f1_score'])
        st.write(f"**Selected Best Model:** {best_model}")
        
        report = eval_results[best_model]['classification_report']
        
        # Format classification report for display
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.4f}"))
