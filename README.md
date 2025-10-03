# Fake News Detector

## Overview

The **Fake News Detector** is a machine learning project designed to classify news articles as either **real** or **fake**. This project uses Python, Jupyter Notebook, and a subset of the **LIAR dataset** for training and testing.  

Fake news can spread quickly online, and this project demonstrates how ML models can help detect misinformation automatically.

---

## Project Structure

FakeNewsProject/
│
├── fakenewsdetector.ipynb # Main Jupyter notebook
├── archive/
│ ├── Fake_subset.csv # Subset of fake news articles
│ └── True_subset.csv # Subset of real news articles
└── README.md # This file

- **Notebook (`.ipynb`)**: Contains all the code, including data cleaning, model training, evaluation, and testing.  
- **CSV files**: Contain news articles with their labels (0 = real, 1 = fake).

---

## Step-by-Step Explanation (Beginner-Friendly)

### 1. Data

The dataset comes from **LIAR**, which contains labeled news statements:

- **`title`**: Headline of the news article  
- **`text`**: Full news content  
- **`label`**: 0 = Real, 1 = Fake  

We only use a **subset of the dataset** for demonstration.

---

### 2. Preprocessing

Before training the model, the text is cleaned:

- Remove punctuation, special characters, and unnecessary spaces  
- Convert all text to lowercase  
- Tokenize and vectorize text for the model  

This ensures the machine learning model can understand the text better.

---

### 3. Splitting Data

We split the dataset into **training** and **testing** sets:

```python
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42
)


Training set: 80% of the data to teach the model

Test set: 20% of the data to evaluate the model

Model Training

We use a machine learning classifier (e.g., Logistic Regression or any text classifier) to train on the cleaned text.

The model learns patterns that distinguish fake vs real news

Training metrics like accuracy, precision, recall, and F1-score are calculated to evaluate performance

Example results from the model:
Accuracy: 0.995
Precision: 0.99
Recall: 1.00
F1-Score: 0.99
Testing / Demo

You can test the model with new sentences:
user_text = "The government has secretly approved a new plan to cancel taxes for everyone."
predict_text(user_text)
Output:

Predicted label → 0 (real) or 1 (fake)

Confidence → How sure the model is about each class
