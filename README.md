# NewsLens AI — BBC News Classifier

A machine learning web application that classifies news articles into five categories using Natural Language Processing (NLP) and a Multinomial Naive Bayes model.

## Overview

NewsLens AI allows users to enter a news article and receive a predicted category along with estimated class probabilities.


## 🚀 Live Demo

**Try NewsLens AI:** [Open the live application](https://newlens-ai-lewzy6n6hwu3ajjck5ikdh.streamlit.app/)

Classify news articles into five categories using Machine Learning and Natural Language Processing (NLP):

- Business
- Entertainment
- Politics
- Sport
- Technology

### ✨ Try It Yourself

1. Open the live application using the link above.
2. Navigate to **Classify News**.
3. Select a sample article or paste your own news article.
4. Click **Classify Article** to get the predicted category.
5. Explore the estimated class probabilities and model evaluation results.


### Supported Categories

- Business
- Entertainment
- Politics
- Sport
- Technology

## Features

- Interactive Streamlit interface
- News article classification
- Predicted category and estimated class probabilities
- Article statistics, including word count and reading time
- Class probability visualization
- Model evaluation metrics and confusion matrix
- Downloadable evaluation report in CSV format

## Tech Stack

- **Language:** Python
- **Machine Learning:** scikit-learn, Multinomial Naive Bayes
- **NLP:** CountVectorizer with unigram and bigram features
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Web Application:** Streamlit
- **Model Persistence:** Joblib

## Model Performance

The trained model's recorded evaluation results are:

| Metric | Score |
|---|---:|
| Test accuracy | 97.65% |
| Macro F1-score | 97.63% |
| Best mean cross-validation macro F1-score | 97.20% |

Sport achieved an F1-score of approximately 99.50% on the recorded test set.

**Evaluation note:** The test set was used in earlier model comparisons. These results may therefore be optimistic and should not be interpreted as an untouched final benchmark or a guarantee of performance on new articles.

## Project Structure

```text
bbc-news-classifier/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── bbc_news_naive_bayes_pipeline.joblib
└── .streamlit/
```

## Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd bbc-news-classifier
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create and activate a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## Limitations

- Predictions depend on the training data and vocabulary learned by the model.
- Articles from unfamiliar domains or with ambiguous content may be misclassified.
- Class probabilities are model estimates, not guaranteed measures of correctness.
- Recorded evaluation results may not represent performance on completely unseen data.

## Future Improvements

- Evaluate the model on a new, untouched test set.
- Add more diverse news sources and articles.
- Compare Naive Bayes with other text classification algorithms.
- Deploy the application to a public hosting platform.

## Author

**Yaswanth Kumar Reddy M**

Machine Learning and Data Science Portfolio Project
