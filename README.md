### Amazon Reviews NLP Analyzer

### 📌 Overview

A simple Python tool that analyzes Amazon product reviews to:

🔍 Extract products and brands mentioned

😊😠 Detect positive/negative sentiment

📊 Generate visual reports

#### 🛠️ How It Works

Input: Takes Amazon reviews (or uses sample data)

1. Processing:

Identifies products/brands (iPhone, Apple, etc.)

Checks if review is positive/negative

2. Output:

Accuracy score

Most mentioned products

Sentiment charts

#### 💻 Quick Start

# 1. Install requirements

- pip install spacy pandas textblob matplotlib seaborn

- python -m spacy download en_core_web_sm

# 2. Run the analyzer
python amazon_reviews_nlp.py

#### 📊 Sample Output
![Results](https://github.com/user-attachments/assets/47594b5a-f12e-4679-8f86-bae8c8c22d2b)


#### 🧠 Key Features

Automatic sample data if no file provided

Simple rule-based sentiment analysis

Visual reporting with matplotlib

Export results to CSV

#### 🚀 Potential Uses

Track product satisfaction

Compare brand reputations

Find common complaints
