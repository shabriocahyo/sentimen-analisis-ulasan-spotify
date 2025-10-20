# Spotify Reviews Sentiment Analysis

## Project Summary
This project performs sentiment analysis on user reviews for the Spotify Android app. It compares three machine learning models — **Logistic Regression**, **Random Forest**, and **Naive Bayes** — to find a good trade-off between accuracy and speed.

## Quick Highlights
- **Dataset:** ~60,000 Spotify reviews (scraped).  
- **Text features:** TF–IDF vectorization.  
- **Labels:** Positive / Neutral / Negative.  
- **Train / Test split:** 75% train, 25% test.

## Main Results
- **Logistic Regression** — Highest accuracy (~78%).  
- **Random Forest** — Good accuracy (~75.4%), slower to train.  
- **Naive Bayes** — Similar accuracy (~75.3%) and the fastest to train/predict.

## Why This Project Matters
App teams can use these findings to pick a model depending on needs:
- Choose **Logistic Regression** for best accuracy.
- Choose **Naive Bayes** for speed and resource efficiency.
- Choose **Random Forest** when robustness and model interpretability matter more than training speed.

## Methods & Processing (simple steps)
1. Collect reviews from Google Play Store (scraped dataset).  
2. Clean text: lowercasing, tokenization, stopwords removal, stemming/lemmatization.  
3. Convert text to numeric features using TF–IDF.  
4. Encode labels (Positive / Neutral / Negative).  
5. Split data into training and testing sets.  
6. Train and evaluate models (accuracy, training time, prediction time, confusion matrix).

## Tools & Libraries
- Python, Jupyter Notebook  
- pandas, numpy  
- nltk (or other text preprocessing libraries)  
- scikit-learn (TF–IDF, models, evaluation)

## How to Reproduce
1. Create a Python environment (Python 3.8+).  
   ```bash
   pip install pandas numpy scikit-learn nltk jupyter
   ```
2. Open the project notebook:  
   ```bash
   jupyter notebook
   ```
3. Load the dataset (CSV/JSON) into the notebook.  
4. Follow the notebook steps: preprocess → TF–IDF → encode → split → train → evaluate.

## Suggested File Structure
```
README.md
data/
  spotify_reviews.csv
notebooks/
  Sentimen_Analisis_Ulasan_Spotify.ipynb
src/
  preprocess.py
  train_models.py
results/
  metrics_summary.md
```

## Tips & Next Steps
- Try modern NLP models (transformers like BERT) to possibly improve accuracy.  
- Use cross-validation and grid search to tune hyperparameters.  
- Monitor sentiment changes across app updates (time series analysis).

## Authors & Credits
- Shabrio Cahyo Wardoyo  
- Muhammad Rizky Saputra  
- Aqsa Idris  
- Zaki Musyaffa Arridha  
Supervised by Umniy Salamah, S.T., MMSI.

## Contact
For questions, use the contact email in the project proposal or the project supervisor's contact.
