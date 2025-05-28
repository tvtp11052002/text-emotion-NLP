# 🧠 Text Emotion Classification using NLP

This project aims to build a system that can classify emotions expressed in user-generated text into five categories: **joy, sadness, anger, fear, neutral**. It combines traditional machine learning, deep learning, and transformer-based models (BERT) to compare performance and improve accuracy in real-world applications such as social media monitoring and customer feedback analysis.

## 📌 Project Overview

With the rapid growth of user-generated content on platforms like social media and e-commerce, identifying the emotional tone behind textual data becomes essential for understanding user behavior. This project presents a complete NLP pipeline for emotion detection — from preprocessing and feature extraction to model training and evaluation.

The system was developed and evaluated on a curated dataset of ~11,000 samples built from three publicly available sources:
- `emotion-stimulus.csv`
- `ISEAR.csv`
- `DailyDialog.csv`

The dataset was balanced and split into training (70%) and test (30%) sets.

## 🚀 Models Implemented

### 🧪 Traditional Machine Learning:
- Naïve Bayes  
- Logistic Regression  
- Random Forest  
- Support Vector Machine (SVM)

### 🤖 Deep Learning:
- LSTM  
- GRU  
→ Both used pretrained **Word2Vec embeddings** (`wiki-news-300d-1M`)

### 🧠 Transformer-based:
- **Fine-tuned BERT** (`bert-base-uncased`) using HuggingFace Transformers

## ✅ Key Results

| Model                 | Accuracy |
|----------------------|----------|
| Naïve Bayes          | 67.02%   |
| Random Forest        | 62.98%   |
| Logistic Regression  | 69.35%   |
| Linear SVM           | 72.71%   |
| GRU + Word2Vec       | 71.77%   |
| **Fine-tuned BERT**  | **82.41%** |

BERT significantly outperformed other models, proving the effectiveness of contextual embeddings and transfer learning.

## ⚙️ Tech Stack

- **Languages**: Python  
- **Libraries**:  
  - NLP: NLTK, spaCy  
  - ML/DL: Scikit-learn, TensorFlow, Keras  
  - Transformers: HuggingFace Transformers  
  - Word Embeddings: Gensim (Word2Vec), FastText  
- **Environment**: Google Colab (T4 GPU), Pandas, Matplotlib, Seaborn

## 📊 Features

- Complete preprocessing pipeline: tokenization, stemming, lemmatization, stopword removal
- Feature engineering: TF-IDF, Word2Vec
- Deep learning model with embedding layers using pretrained vectors
- Fine-tuning transformer-based model (BERT) for text classification
- Evaluation metrics: Accuracy, F1-score, Confusion Matrix, Loss/Accuracy curves

## 🧰 How to Run

1. Clone the repository  
   ```bash
   git clone https://github.com/yourusername/text-emotion-classification.git
   cd text-emotion-classification
