```markdown
# 🧠 Text Emotion Classification using NLP

A comprehensive NLP project to classify human emotions from text (joy, sadness, anger, fear, neutral) using a combination of traditional machine learning, deep learning, and transformer-based models (BERT). The project includes full preprocessing pipelines, multiple model implementations, and performance comparisons.

## 📂 Project Structure

```

text-emotion-nlp/
│
├── finetuned_BERT.ipynb              # Notebook for fine-tuning BERT model
├── lstm.ipynb                  # Notebook for building LSTM/GRU models
├── traditional_ml.ipynb        # Notebook for traditional ML models (SVM, LR, NB, RF)
├── model.png                   # Architecture diagram (optional)
│
├── data/                       # Processed training/testing data
│   ├── train.csv
│   ├── test.csv
│   └── datasets/               # Original datasets
│       ├── dailydialog.csv
│       ├── emotion-stimulus.csv / .xlsx
│       ├── isear.csv / .xlsx
│
├── docx/                       # Final report (PDF + DOCX)
│   ├── ProjectNLP.docx
│   └── ProjectNLP.pdf
│
├── embeddings/                 # Placeholder for pretrained embeddings
│   └── .gitignore
│
├── models/                     # Saved models
│   ├── tfidf\_svm.sav           # Serialized SVM model with TF-IDF
│   └── emotion\_detector/       # Fine-tuned BERT saved model (TensorFlow format)
│       ├── saved\_model.pb
│       ├── keras\_metadata.pb
│       ├── fingerprint.pb
│       ├── assets/
│       └── variables/

````

## 📌 Overview

This project classifies user-generated texts into five emotional categories. It includes data preprocessing, feature extraction using TF-IDF and Word2Vec, and model training with ML algorithms (SVM, NB, etc.), RNN architectures (LSTM, GRU), and transfer learning using fine-tuned BERT. The best model was fine-tuned BERT with over **82% accuracy**.

## ✅ Key Achievements

- Fine-tuned **BERT** model achieved **82.41% accuracy** — best among all evaluated models.
- Built GRU model with pretrained FastText embeddings, reaching **71.77% accuracy**.
- Evaluated multiple approaches: Naïve Bayes, Logistic Regression, Random Forest, SVM, LSTM, GRU, BERT.
- Full NLP pipeline implemented: text cleaning, tokenization, TF-IDF vectorization, Word2Vec embedding.
- Trained models on **Google Colab (GPU T4)** with proper visualization and evaluation metrics.

## ⚙️ Tech Stack

- **Languages**: Python  
- **Libraries**: NLTK, spaCy, Scikit-learn, TensorFlow, Keras, HuggingFace Transformers, Gensim  
- **Tools**: Google Colab, Matplotlib, Seaborn  
- **Models**: Naïve Bayes, Logistic Regression, Random Forest, SVM, LSTM, GRU, BERT  
- **Embeddings**: TF-IDF, Word2Vec (wiki-news-300d-1M)

## 📊 Features

- Dataset preprocessing: cleaning, lemmatization, tokenization
- Feature extraction: TF-IDF and Word2Vec
- ML and DL model training and evaluation
- BERT fine-tuning with HuggingFace Transformers
- Emotion prediction on real-time test inputs
- Saved models for deployment (`/models/`)

## 🧰 How to Run

1. Clone the repository  
   ```bash
   git clone https://github.com/yourusername/text-emotion-nlp.git
   cd text-emotion-nlp
````

2. (Optional) Set up virtual environment and install requirements

   ```bash
   pip install -r requirements.txt
   ```

3. Open notebooks:

   * `traditional_ml.ipynb` → train and test ML models
   * `lstm.ipynb` → build and evaluate GRU/LSTM models
   * `finetuned BERT.ipynb` → fine-tune BERT with HuggingFace and TensorFlow

4. Test saved models from `/models/` folder on new messages.

## 👨‍💻 Authors

* Trần Văn Tuấn Phong
*(Supervised by Đoàn Thị Hồng Phước – Hue University)*
