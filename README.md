# Sentiment-Fake-News-Analysis
Supervised Learning Course Project

## Overview
This project builds a complete **Natural Language Processing (NLP) pipeline** for:

- Sentiment Analysis (Positive / Negative)
- Fake News Detection (future work)

The system combines:
- Classical Machine Learning
- Deep Learning (CNN, Autoencoder)
- Recurrent Networks (GRU, **LSTM**, Vanilla RNN)
- **Fine-tune BERT** (planned next step — see Phase 3 → Future Work)

---

## Datasets
- **Sentiment140** → Twitter Sentiment Analysis (1.6M tweets)
  - Labels: `0` = Negative, `4` = Positive (remapped to `0` / `1`)
  - Columns: `target`, `id`, `date`, `flag`, `user`, `text`

---

## Pipeline Structure

The project is divided into **3 main phases**:

---

# Phase 1 — Data Processing & Classical ML

### Goal
Build a strong baseline using traditional NLP techniques.

### Tasks
- **Data loading & label remapping** — load Sentiment140 (1.6M tweets) and remap labels (`0 → Negative`, `4 → Positive (1)`)
- **Text cleaning pipeline:**
  - Lowercasing
  - Removing URLs, `@mentions`, `#hashtags`, and HTML entities
  - Removing non-alphabetic characters
  - Tokenization (NLTK)
  - Stopword removal (NLTK English stopwords)
  - Stemming with the **Porter Stemmer**
- **Duplicate removal** (~18K duplicate tweets dropped)
- **Feature engineering:**
  - Tweet length (characters)
  - Word count
  - Clean word count (post-preprocessing)
- **Exploratory Data Analysis (EDA):**
  - Sentiment class balance
  - Tweet-length distribution
  - Top words per class (Negative vs Positive)
  - Tweet-time analysis (hour-of-day)
  - Correlation between numeric features and sentiment
- **Feature extraction with TF-IDF** (`ngram_range=(1, 2)`) producing a sparse high-dimensional matrix
- **Dimensionality reduction with TruncatedSVD / LSA** (100 components) — required for KNN / Random Forest / XGBoost to be tractable on 1.6M tweets
- **Classical model training & evaluation:**
  - **K-Nearest Neighbours** (on SVD features) — Accuracy ~0.6511
  - **Multinomial Naïve Bayes** (on TF-IDF directly) — Accuracy ~0.75 (best Phase 1 model)
  - **Random Forest** (on SVD features) — Accuracy ~0.70
  - **XGBoost** (on SVD features) — Accuracy ~0.70
- **Hyperparameter tuning** with `RandomizedSearchCV` (chosen over `GridSearchCV` for speed at this scale)
- **Model comparison** — Accuracy, F1, Precision, Recall, and confusion matrices across all four classifiers

---

# Phase 2 — Deep Learning (CNN + Autoencoder)

### Goal
Improve performance using neural networks.

### Tasks
- Tokenization + padding pipeline (custom word→id vocabulary, post-padding to a fixed `SEQ_MAX_LEN`)
- **1D CNN** for sentiment classification (`Embedding → Conv1D → GlobalMaxPooling1D → Dense → Dropout → Sigmoid`)
- **Autoencoder** for unsupervised feature learning / anomaly detection
- Pre-trained embeddings:
  - **GloVe 200d** (frozen and trainable variants)
  - Random initialization (no pre-trained vectors) as a baseline
- Hyperparameter experiments:
  - **Dropout sweep** (0.3 vs 0.5)
  - **Embedding strategy** (GloVe frozen vs GloVe trainable vs random init)
  - **Optimizer comparison** (Adam vs RMSprop vs SGD + Momentum)
- Early stopping (`patience=3`, `restore_best_weights=True`)

---

# Phase 3 — Recurrent Models

### Goal
Compare gated and non-gated recurrent architectures against the CNN baseline.

### Completed
- **GRU (GloVe frozen, Adam)** — `Embedding → GRU(128) → Dropout → Dense → Sigmoid`
  - Accuracy = **0.7613**, F1 = **0.7664**
  - Best frozen-GloVe model on the leaderboard.
- **LSTM (GloVe trainable, Adam + ReduceLROnPlateau)** — `Embedding → LSTM(64) → Dropout → Dense → Dropout → Sigmoid`
  - Accuracy = **0.7824**, F1 = **0.7793**
  - Best recurrent model overall; runner-up on the global leaderboard.
  - Uses `EarlyStopping(patience=3)` + `ReduceLROnPlateau(factor=0.5, patience=1)`; stopped after 5 epochs (best `val_loss` at epoch 2).
- **Vanilla RNN (`SimpleRNN`, GloVe frozen, Adam)** — same depth as the GRU; only the recurrent cell changes
  - Accuracy = **0.5017**, F1 = **0.6682**
  - Collapsed to the majority class (vanishing gradients + frozen embeddings); included as a textbook demonstration of why gated cells are preferred.

### Future Work

#### Fine-tune BERT
The natural next step beyond the recurrent and convolutional baselines is to fine-tune a pre-trained transformer.

- **Model:** `bert-base-uncased` (110M parameters) — alternative: `distilbert-base-uncased` for faster iteration on Sentiment140 (1.6M tweets).
- **Tokenizer:** Hugging Face `AutoTokenizer` with `max_length = 64` (tweets are short).
- **Head:** Sequence-classification head with a single sigmoid output (binary sentiment).
- **Training plan:**
  - Optimizer: AdamW, learning rate = `2e-5`, weight decay = `0.01`
  - Scheduler: linear warmup + linear decay
  - Epochs: 2–3 (transformers fine-tune quickly on this much data)
  - Batch size: 32 with gradient accumulation if needed
  - Mixed precision (`fp16`) to fit on a single consumer GPU
- **Target metrics:** ≥ 0.82 accuracy / F1, surpassing the current best (CNN trainable GloVe at 0.7825).
- **Comparison hook:** the trained model will be registered in `all_models` under `"BERT (fine-tuned)"` so it slots into the existing leaderboard, ROC plots, and confusion-matrix grid in Section 10.
- **Dependencies (when implemented):** `transformers >= 4.40`, `accelerate` (for `fp16` + multi-GPU), `datasets` (optional, for streaming).

#### Other directions
- Bi-LSTM / stacked LSTM
- Attention mechanism over GRU / LSTM outputs
- Fake News Detection dataset

---

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve & AUC
- Confusion Matrix

---

## Technologies Used
- Python 3.11
- Scikit-learn
- XGBoost
- NLTK
- Pandas / NumPy
- Matplotlib / Seaborn
- TensorFlow / Keras

---

## Results Summary

Test-set results from `sentiment.ipynb` (Sentiment140 hold-out, 316,294 tweets). Phase 1 numbers are approximate (per the original report); Phase 2 / 3 numbers are exact from the notebook outputs.

| Phase | Model                                   | Accuracy | F1     |
|-------|-----------------------------------------|---------:|-------:|
| 1     | KNN (TruncatedSVD, k tuned)             | ~0.6511  | ~0.6532 |
| 1     | Naïve Bayes (Multinomial, α tuned)      | ~0.75    | ~0.75   |
| 1     | Random Forest (on SVD features)         | ~0.70    | ~0.70   |
| 1     | XGBoost (on SVD features)               | ~0.70    | ~0.70   |
| 2     | CNN — GloVe frozen, Adam, drop=0.5      | 0.7475   | 0.7487  |
| 2     | CNN — GloVe frozen, Adam, drop=0.3      | 0.7482   | 0.7538  |
| 2     | CNN — GloVe frozen, RMSprop             | 0.7482   | 0.7547  |
| 2     | CNN — GloVe frozen, SGD + Momentum      | 0.7462   | 0.7512  |
| 2     | CNN — GloVe trainable, Adam, drop=0.5   | **0.7816** | **0.7825** |
| 2     | CNN — No pre-trained Emb (random init)  | 0.7789   | 0.7820  |
| 3     | **LSTM** (GloVe trainable, Adam + RLR)  | **0.7824** | 0.7793  |
| 3     | **GRU** (GloVe frozen, Adam)            | 0.7613   | 0.7664  |
| 3     | SimpleRNN (GloVe frozen, Adam)          | 0.5017   | 0.6682  |
| —     | _Fine-tune BERT_ (planned next step)    | _target ≥ 0.82_ | _target ≥ 0.82_ |

**Overall best (F1):** CNN with trainable GloVe embeddings (Acc = 0.7816, F1 = 0.7825).
**Overall best (Accuracy):** LSTM with trainable GloVe (Acc = 0.7824, F1 = 0.7793).
**Best frozen-GloVe model:** GRU (Acc = 0.7613, F1 = 0.7664).

---

## Key Insights
- **Phase 1:** Naïve Bayes is the strongest classical baseline on TF-IDF features; KNN is slow and does not scale; ensemble models help robustness but need tuning; dimensionality reduction (TruncatedSVD) is essential for KNN / RF / XGB to be tractable on 1.6M tweets.
- **Phase 2:** CNNs clearly beat the classical baselines (+~13 accuracy points). Allowing the embedding layer to fine-tune is the single most impactful change — trainable GloVe and random init perform nearly identically and both significantly beat frozen GloVe. Optimizer choice has only a marginal effect when embeddings are frozen.
- **Phase 3:** Gating is what makes recurrence work here — swapping `SimpleRNN` for `GRU` lifts accuracy by ~26 points with everything else fixed. The GRU beats every frozen-GloVe CNN configuration on F1, suggesting that sequential context is valuable when the embedding layer cannot adapt. The **LSTM with trainable GloVe** reaches the best accuracy of any model trained so far (0.7824), confirming that the *trainable-embedding + gated-recurrence* combination matches the CNN trainable-GloVe configuration on its own terms. The SimpleRNN failure is a textbook vanishing-gradient case and is left in the notebook for pedagogical reasons.
- **Next step (planned):** Fine-tuning **BERT** is the natural next step; transformer-based contextual embeddings should comfortably exceed the 0.78 ceiling reached by the trainable-GloVe + recurrence models.
- **Engineering:** Early stopping (`patience=3`, `restore_best_weights=True`) is effective at preventing overfitting and saves substantial GPU time across all DL models. `ReduceLROnPlateau` further stabilises LSTM training once `val_loss` plateaus.

---

## Repository Layout
- `sentiment.ipynb` — main notebook covering all three completed phases end-to-end (Sections 1–13).
- `Report.pdf` — written report (17 pages, unified theme): title page, Phase 1, Phase 2, and Phase 3 (GRU + LSTM + Vanilla RNN).
- `requirements.txt` — Python dependencies.
- `training.1600000.processed.noemoticon.csv` — Sentiment140 dataset (not version-controlled).
- `glove.6B.200d.txt` — pre-trained GloVe vectors used by the CNN / GRU / LSTM / RNN models (not version-controlled).

---

## Setup

```bash
pip install -r requirements.txt
```

Then open `sentiment.ipynb` and run the cells in order. The deep learning sections (Phase 2 and Phase 3) expect a CUDA-capable GPU for reasonable training times — the GRU section trains at ~220 s/epoch on GPU, the LSTM at ~250 s/epoch.

When the Fine-tune BERT experiment (see Phase 3 → Future Work) is added, install the transformer stack on top of the current environment:

```bash
pip install "transformers>=4.40" "accelerate>=0.30" "datasets>=2.18"
```
