## Notebooks Overview

This folder contains the Jupyter notebooks used for data preparation, baseline sentiment evaluation, and machine learning–based sentiment classification on Indian financial news headlines.

--- 
### Note: 
This project uses spaCy for text preprocessing. Install the English model using: 
```python
python -m spacy download en_core_web_sm
```

### 01_data_preprocessing.ipynb
- Loads the raw Indian financial news headlines dataset.
- Performs data cleaning and filtering, including:
  - Removal of non-English / noisy text
  - Date normalization
  - Limiting headlines to a maximum of 60 per day
- Produces a refined dataset (`eng_top60.csv`) with balanced positive and negative sentiment labels.
- Serves as the foundational dataset for all downstream modeling.

---

### 02_vader_sentiment_baseline.ipynb
- Applies the VADER lexicon-based sentiment analyzer to financial news headlines.
- Converts compound sentiment scores into binary sentiment labels.
- Evaluates performance using accuracy, confusion matrix, and classification report.
- Acts as a benchmark to compare lexicon-based sentiment analysis against ML models.

---

### 03_ml_models_tfidf.ipynb
- Preprocesses text using spaCy (tokenization and lemmatization).
- Transforms text into numerical features using TF-IDF vectorization.
- Trains and evaluates multiple supervised ML models:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Linear Support Vector Classifier (LinearSVC)
- Compares model performance using accuracy and classification metrics.

---

### 04_model_tuning_and_ensemble.ipynb
- Performs hyperparameter tuning using GridSearchCV on top-performing models.
- Builds a Voting Classifier ensemble combining multiple tuned classifiers.
- Evaluates the ensemble model against individual models.
- Demonstrates incremental performance improvements through ensembling and optimization.
