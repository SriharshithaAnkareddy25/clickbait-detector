# Clickbait Headline Detector

A machine learning project that detects clickbait news headlines using NLP techniques.
Built as part of a Programming for Data Science course project.

## Dataset
- **Source:** News Headlines Dataset for Sarcasm Detection (Kaggle)
- **Size:** 26,709 headlines (after cleaning: 26,602)
- **Labels:** 0 = Not Clickbait, 1 = Clickbait

## Project Pipeline
1. Data Collection & Loading
2. Data Cleaning (null check, duplicate removal)
3. Text Preprocessing (lowercasing, stopword removal, lemmatization)
4. Exploratory Data Analysis (class distribution, word clouds, headline length)
5. Feature Extraction (TF-IDF with unigrams and bigrams)
6. Model Training (Logistic Regression, SVM, Random Forest)
7. Evaluation (Accuracy, F1 Score, Confusion Matrix)
8. Live Demo (predict any headline)

## Results

| Model               | Accuracy | F1 Score |
|---------------------|----------|----------|
| Logistic Regression | 78.54%   | 73.99%   |
| SVM                 | 78.37%   | 74.86%   |
| Random Forest       | 76.00%   | 71.46%   |

## Key Findings
- Clickbait headlines tend to use vague words like "man", "area", "nation" to create curiosity
- Non-clickbait headlines use specific names and facts like "Trump", "federal reserve"
- SVM achieved the best F1 score making it the most reliable clickbait detector
- Model limitations stem from dataset bias (HuffPost vs The Onion)

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/clickbait_detection.ipynb
```

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (Logistic Regression, SVM, Random Forest)
- NLTK (stopwords, lemmatization)
- Matplotlib, Seaborn, WordCloud

## References
Based on survey paper: *Beyond the Headline: An In-Depth Survey of Multimodal Clickbait
Detection within the Indian Digital Ecosystem*