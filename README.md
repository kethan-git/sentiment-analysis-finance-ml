# Sentiment Analysis on Financial Data

### Short description
Machine learning + NLP pipeline that performs sentiment classification on Indian financial news headlines (TF-IDF → ML models → hyperparameter tuning → voting ensemble). 

---

## TL;DR
- **Input:** Indian financial news headlines (2017–2021) from a Kaggle dataset (pre-scraped). 
- **Approach:** text cleaning + spaCy preprocessing → TF-IDF vectorization → train/test on ML models (Logistic Regression, LinearSVC, MultinomialNB) → GridSearchCV → Voting Classifier ensemble. 
- **Best achieved:** ensemble accuracy ≈ 77.18% (individual models: Logistic Regression ≈ 76.62%, LinearSVC ≈ 76.52%). VADER lexicon baseline ≈ 61.10%. 
--- 

## Why this project
News headlines can meaningfully influence short-term market sentiment. 
This project demonstrates end-to-end steps to turn raw headlines into actionable sentiment labels using classical ML techniques and quantifies where those techniques succeed or fail on financial text. 

---

## Dataset
- **Name / source:** *Indian Financial News Headlines Sentiments* (Kaggle; scraped using GDELT headline scraper by Harsh Khandelwal). 
- **Original size:** ~200,498 rows (2017–04/2021). 
- **Refined dataset used in experiments:** 89,663 rows after filtering and capping at 60 headlines/day; class split ~47% positive / 53% negative. 
---

## Methods & Pipeline 
- **Data cleaning & filtering:** remove non-ASCII/Hindi garbage, normalize date format, cap headlines/day. 
- **Text preprocessing with spaCy:** tokenization, lemmatization, basic stopword handling. 
- **Feature extraction:** TF-IDF vectorizer (tuned max_features, ngram_range where useful). 
- **Baseline lexicon test:** VADER (rule/lexicon-based) for comparison. 
- **Supervised models:** MultinomialNB, LinearSVC, Logistic Regression (pipelines combining TF-IDF + classifier). 
- **Hyperparameter tuning:** GridSearchCV (5-fold) on TF-IDF + classifier hyperparameters. 
- **Ensemble methods:** VotingClassifier (hard voting) combining the tuned base estimators.
---

## Results Summary
| Model | Accuracy |
|------|----------|
| VADER (Baseline) | 61.10% |
| Multinomial NB | 74.20% |
| LinearSVC | 77.22% |
| Logistic Regression | 77.21% |
| Voting Classifier | 77.18% |

**Interpretation:** Classical ML with hyperparameter tuning via GridSearchCV on TF-IDF and classifier hyperparameters, outperforming general-purpose lexicon tools (VADER), but domain-specific language and context limit top-end accuracy - motivating transformer-based and domain-adapted models (FinBERT) for future work.
