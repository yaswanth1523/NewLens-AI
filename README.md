# 📰 NewsLens AI — BBC News Article Classifier

An end-to-end Machine Learning application that classifies BBC news articles into five categories using Natural Language Processing (NLP) and Multinomial Naive Bayes. Built with Scikit-learn and Streamlit, the application provides real-time predictions, estimated class probabilities, and model evaluation insights through an interactive web interface.

## 🚀 Live Demo

**Streamlit App:** [Open NewsLens AI](https://newlens-ai-lewzy6n6hwu3ajjck5ikdh.streamlit.app/)

> Replace the URL above with your actual deployed Streamlit Community Cloud URL.

## 📌 Project Overview

NewsLens AI automatically categorizes news articles based on their textual content. The project demonstrates an end-to-end text classification workflow, from data preparation and model evaluation to deployment as an interactive web application.

The model predicts one of five categories:

- Business
- Entertainment
- Politics
- Sport
- Technology

## ✨ Features

- **News Classification:** Predicts the category of a news article from its text.
- **Sample Articles:** Test the application using sample news articles.
- **Class Probabilities:** Displays the model's estimated probability for each category.
- **Text Statistics:** Shows word count, character count, and estimated reading time.
- **Model Evaluation:** Displays accuracy, precision, recall, and F1-score.
- **Category-Level Analysis:** Provides a per-category performance report.
- **Confusion Matrix:** Visualizes correct predictions and classification errors.
- **Downloadable Reports:** Download model evaluation results and confusion matrix data as CSV files.
- **Interactive Interface:** Navigate between Home, Classify News, and Model Information pages.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Model training, preprocessing, and evaluation |
| CountVectorizer | Converts text into numerical features |
| Multinomial Naive Bayes | News classification algorithm |
| Matplotlib | Data visualization |
| Streamlit | Interactive web application |
| Joblib | Saving and loading the trained model |
| Git and GitHub | Version control and project hosting |
| Streamlit Community Cloud | Application deployment |

## 🧠 Machine Learning Workflow

1. **Data Collection:** Loaded the BBC news dataset containing 2,224 articles.
2. **Data Cleaning:** Removed duplicate records, resulting in 2,126 articles.
3. **Text Preprocessing:** Prepared article text for feature extraction.
4. **Feature Extraction:** Used CountVectorizer with unigram and bigram features.
5. **Train-Test Split:** Divided the dataset into 1,700 training records and 426 testing records.
6. **Model Selection:** Evaluated Multinomial Naive Bayes configurations using five-fold cross-validation.
7. **Hyperparameter Tuning:** Selected `ngram_range=(1, 2)` and `alpha=0.1`.
8. **Model Evaluation:** Evaluated classification performance using accuracy, precision, recall, F1-score, and a confusion matrix.
9. **Model Persistence:** Saved the trained pipeline using Joblib.
10. **Deployment:** Integrated the saved pipeline into a Streamlit application and deployed it to Streamlit Community Cloud.

## 📊 Dataset Information

| Metric | Value |
|---|---:|
| Original articles | 2,224 |
| Articles after duplicate removal | 2,126 |
| Training records | 1,700 |
| Testing records | 426 |
| Number of categories | 5 |

### Category Distribution

| Category | Articles |
|---|---:|
| Sport | 504 |
| Business | 503 |
| Politics | 403 |
| Entertainment | 369 |
| Technology | 347 |
| **Total** | **2,126** |

## 📈 Model Performance

The selected model achieved the following results on the test dataset.

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 97.65% |
| Macro Precision | 97.60% |
| Macro Recall | 97.69% |
| Macro F1-score | 97.63% |
| Best mean cross-validation Macro F1-score | 97.20% |

### Per-Category Performance

| Category | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Business | 96.97% | 95.05% | 96.00% |
| Entertainment | 100.00% | 97.30% | 98.63% |
| Politics | 95.29% | 100.00% | 97.59% |
| Sport | 100.00% | 99.01% | 99.50% |
| Technology | 95.71% | 97.10% | 96.40% |

**Interpretation:** The model correctly classified 416 out of 426 test articles. Its overall results indicate strong performance across all five categories, with some remaining classification errors between business, politics, and technology articles.

**Evaluation limitation:** The test set was also used in earlier model comparisons during development. Consequently, the reported test performance may be optimistic and should not be interpreted as a completely independent final estimate of generalization performance. A fresh holdout set would provide a stronger assessment.

## 📁 Project Structure

```text
NewLens-AI/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│   └── bbc_news_naive_bayes_pipeline.joblib
│
├── evaluation/
│   ├── bbc_news_evaluation_report.csv
│   └── bbc_news_confusion_matrix.csv
│
└── .streamlit/
```

## ⚙️ Run the Project Locally

### Prerequisites

- Python 3.12
- Git
- pip

### 1. Clone the repository

```bash
git clone https://github.com/yaswanth1523/NewLens-AI.git
cd NewLens-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows — Command Prompt:**

```cmd
.venv\Scripts\activate
```

**Windows — PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application should open in your browser. If it does not, open the local URL displayed in your terminal, usually `http://localhost:8501`.

## 📦 Saved Model

The trained pipeline is stored at:

```text
models/bbc_news_naive_bayes_pipeline.joblib
```

The pipeline combines CountVectorizer text feature extraction with Multinomial Naive Bayes classification. Saving the pipeline allows the application to load the trained model and make predictions without retraining it every time the application starts.

**Important:** The serialized model should be loaded in a compatible Python and Scikit-learn environment. The project pins dependency versions in `requirements.txt` to support reproducibility.

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud** and hosted in the GitHub repository.

To deploy your own copy:

1. Push the project files to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Create a new app and select the repository, branch, and `app.py` entry point.
4. Deploy the application.
5. Test the classification page, evaluation metrics, charts, and CSV downloads.

## 🔮 Future Improvements

- Evaluate the model using a fresh, untouched test dataset.
- Compare Multinomial Naive Bayes with Logistic Regression, Linear SVM, and other text classification models.
- Explore TF-IDF features and more advanced NLP techniques.
- Add confidence interpretation and guidance for short or ambiguous articles.
- Expand the application to support additional news categories.
- Add automated tests and continuous integration for improved reliability.

## 👨‍💻 Author

**Yaswanth Kumar Reddy M**

- **GitHub:** [yaswanth1523](https://github.com/yaswanth1523)
- **Project Repository:** [NewLens-AI](https://github.com/yaswanth1523/NewLens-AI)
- **LinkedIn:** [Add your LinkedIn profile URL](https://www.linkedin.com/)
