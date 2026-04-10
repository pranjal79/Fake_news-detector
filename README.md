

## 📌 Project Overview

This is a **production-level Fake News Detection system** built using NLP and MLOps practices.

The project detects whether a news article is **Fake (0)** or **Real (1)** using machine learning models trained on a Kaggle dataset.

### 🔥 Key Features:

* End-to-end ETL pipeline
* NLP preprocessing (cleaning, tokenization, stopword removal)
* TF-IDF feature engineering
* Multiple model training & comparison
* Experiment tracking using MLflow + DagsHub
* Data versioning using DVC
* Fully reproducible pipeline (`dvc repro`)

---

## 🗂️ Project Structure

```
fake_news_detector/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── features/
├── models/
├── reports/
├── src/
│   ├── etl/
│   ├── features/
│   ├── models/
│   └── utils/
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

Fake and Real News Dataset (Kaggle):
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

| File     | Records | Label |
| -------- | ------- | ----- |
| Fake.csv | ~23,000 | 0     |
| True.csv | ~21,000 | 1     |

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```
git clone https://github.com/pranjal79/Fake_news-detector.git
cd Fake_news-detector
```

---

### 2. Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

### 4. Download NLTK Data

```
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

### 5. Setup Environment Variables

Create `.env` file:

```
DAGSHUB_USERNAME=your_username
DAGSHUB_TOKEN=your_token
MLFLOW_TRACKING_URI=https://dagshub.com/pranjal79/Fake_news-detector.mlflow
```

---

### 6. Start MongoDB

```
net start MongoDB
```

---

## 🚀 Run the Pipeline

### Run full pipeline:

```
dvc repro
```

---

## 🧪 Models Used

| Model               | Accuracy  | F1 Score   |
| ------------------- | --------- | ---------- |
| Logistic Regression | 0.993     | 0.993      |
| Naive Bayes         | 0.956     | 0.954      |
| SVM                 | **0.997** | **0.9968** |

🏆 **Best Model: SVM**

---

## 📊 Metrics

View metrics using:

```
dvc metrics show
```

---

## 🔁 Data Versioning

```
dvc pull
dvc push
dvc status
```

---

## 🗄️ MongoDB

Database: `fake_news_db`

Collections:

* `raw_news`
* `processed_news`

---

## 📈 Experiment Tracking

Tracked using MLflow + DagsHub:
https://dagshub.com/pranjal79/Fake_news-detector

---

## 🛠️ Tech Stack

* Python
* NLTK
* Scikit-learn
* MongoDB
* MLflow
* DVC
* DagsHub

---

## 👤 Author

**Pranjal Panigrahi**

* GitHub: https://github.com/pranjal79
* DagsHub: https://dagshub.com/pranjal79

---

## 🎯 Project Highlights

* Built a complete MLOps pipeline
* Automated workflow using DVC
* Achieved ~99.7% accuracy using SVM
* Integrated experiment tracking with MLflow
* Used MongoDB for scalable data storage

---

## 🚀 Future Improvements

* Deploy using Streamlit / FastAPI
* Add deep learning models (LSTM, BERT)
* Real-time news prediction API

---

⭐ If you like this project, give it a star!
