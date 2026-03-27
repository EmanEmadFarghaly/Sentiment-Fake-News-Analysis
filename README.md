# Sentiment-Fake-News-Analysis
Supervised Learning Course Project

## 📖 Overview
This project builds a complete **Natural Language Processing (NLP) pipeline** for:

- Sentiment Analysis (Positive / Negative)
- Fake News Detection

The system combines:
- Classical Machine Learning
- Deep Learning
- Transformer-based Models

---

## 📊 Datasets
- **Sentiment140** → Twitter Sentiment Analysis (1.6M tweets)
  
---

## ⚙️ Pipeline Structure

The project is divided into **3 main phases**:

---

# 🟢 Phase 1 — Data Processing & Classical ML

### 🎯 Goal
Build a strong baseline using traditional NLP techniques

---

### Malak Mohamed — Data Cleaning & EDA & Visualization
- Data loading and inspection  
- Handling duplicates & missing values  
- Text preprocessing:
  - Lowercasing  
  - Removing URLs, mentions, hashtags  
  - Stopwords removal  
  - Stemming  
- Feature engineering:
  - Tweet length  
  - Word counts  

---

### Mohamed Beshr  —  Text Preprocessing & TF-IDF
- Sentiment distribution (bar + pie)  
- Tweet length analysis  
- Top frequent words  
- Time-based analysis (hour/month)  
- Correlation heatmap  
- User activity analysis  
- Visualizations (Matplotlib + Seaborn)  
- TF-IDF feature extraction 

---

### Essam Mamdouh — Classical Models (KNN + Naïve Bayes)
 - Train/test split  
- KNN with dimensionality reduction (TruncatedSVD)  
- Naïve Bayes classifier  
- Hyperparameter tuning:
  - K selection  
  - Alpha smoothing  
- Model evaluation:
  - Accuracy, Precision, Recall, F1  
  - Confusion Matrix  

---

### Eman Emad — Ensemble Models & Optimization
- Dimensionality reduction (TruncatedSVD)  
- Random Forest classifier  
- XGBoost classifier  
- Hyperparameter tuning (RandomizedSearchCV)  
- Model comparison (KNN, NB, RF, XGB)  
- ROC Curve visualization  
- Final evaluation & insights  

---

# 🔵 Phase 2 — Deep Learning

### 🎯 Goal
Improve performance using neural networks

### Tasks:
- Train **1D CNN** for text classification  
- Build **Autoencoder** for anomaly detection  
- Use pretrained embeddings:
  - GloVe  
  - FastText  
- Apply:
  - Dropout  
  - Optimizers tuning  

---

# 🟣 Phase 3 — Advanced Models

### 🎯 Goal
Use state-of-the-art NLP models

### Tasks:
- LSTM / Bi-LSTM models  
- Attention mechanism  
- Fine-tune:
  - BERT  
  - DistilBERT  
- Model explainability:
  - SHAP  
  - LIME  
- GAN:
  - Generate adversarial fake news samples  

---

## 📈 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC Curve & AUC  

---

## 🚀 Technologies Used
- Python  
- Scikit-learn  
- XGBoost  
- NLTK  
- Pandas / NumPy  
- Matplotlib / Seaborn  
- TensorFlow / PyTorch  

---

## 📊 Results Summary (Example)

| Model            | Accuracy | F1 Score |
|------------------|---------|---------|
| KNN              | ~65%    | ~0.65   |
| Naïve Bayes      | ~75%    | ~0.75   |
| Random Forest    | ~70%    | ~0.70   |
| XGBoost          | ~70%    | ~0.70   |

---

## 🧠 Key Insights
- Naïve Bayes performs best on TF-IDF features  
- KNN is slower and less scalable  
- Ensemble models improve robustness but require tuning  
- Dimensionality reduction improves efficiency  

---


```bash
pip install -r requirements.txt
