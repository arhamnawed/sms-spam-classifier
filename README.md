# SMS Spam Detection using Machine Learning

## Overview
This project builds a machine learning pipeline to classify SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) techniques.

The notebook performs:
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Text preprocessing using NLP
- Feature extraction with TF-IDF and Bag-of-Words
- Training and evaluation of multiple machine learning models
- Ensemble learning comparison
- Saving the final trained model and vectorizer

---

# Dataset
The project combines two SMS spam datasets:

- `spam_ham_india.csv`
- `cleaned_sms_spam.csv`

## Dataset Processing
- Renamed columns:
  - `Msg` → `text`
  - `Label` → `target`
- Encoded labels using `LabelEncoder`
  - `ham = 0`
  - `spam = 1`
- Removed invalid numerical rows from the text column
- Merged both datasets into a single dataframe

---

# Libraries Used

```python
pandas
numpy
matplotlib
seaborn
nltk
scikit-learn
wordcloud
pickle