# -Sentiment-Analysis-Using-Machine-Learning-Arabic-Text-
This project focuses on Sentiment Analysis using Machine Learning (ML) techniques to classify Arabic text into positive, negative, or neutral sentiments. It builds upon previous NLP preprocessing techniques and applies ML models for sentiment prediction.

📊 Overview
With the increasing use of social media platforms, sentiment analysis has become an essential tool for understanding public opinion. This project preprocesses an Arabic tweets dataset and applies TF-IDF vectorization to transform text into numerical features suitable for machine learning classification.

📂 Dataset
	•	The dataset consists of Arabic tweets labeled as Positive, Negative, or Neutral.
	•	Preprocessing Steps:
	•	Remove non-Arabic characters, diacritics (Tashkeel), and Tatweel (ـ).
	•	Normalize text and remove stopwords.
	•	Convert text to TF-IDF (Term Frequency-Inverse Document Frequency) features.

 🛠 Methodology

1️⃣ Data Preprocessing
	•	Tokenization & Text Cleaning
	•	Removing Special Characters, Diacritics & Stopwords
	•	TF-IDF Vectorization for Feature Extraction

2️⃣ Machine Learning Model: Random Forest Classifier
	•	Why Random Forest?
	•	Robustness: Reduces overfitting by averaging multiple decision trees.
	•	Scalability: Handles large datasets effectively.
	•	Feature Importance: Identifies key words affecting sentiment classification.

3️⃣ Model Performance
	•	Accuracy: 61.23%
	•	Key Observations:
	•	Neutral sentiment had the highest recall (80%).
	•	Positive sentiment had low recall (4%), meaning the model struggles with positive sentiment detection.
	•	Hyperparameter Tuning:
	•	Initial Parameters: n_estimators=100, max_depth=20 → 59% Accuracy
	•	Optimized Parameters: n_estimators=300, max_depth=50 → 61% Accuracy


 📢 Recommendations for Improvement

To improve sentiment classification accuracy, consider:
✅ Using More Suitable Classifiers:
	•	Logistic Regression (best suited for TF-IDF-based text classification).
	•	Support Vector Machines (SVM) (effective for high-dimensional text data).

✅ Replacing TF-IDF with Word Embeddings:
	•	Word2Vec, FastText, or BERT for better semantic understanding.

✅ Handling Class Imbalance:
	•	Implement class_weight="balanced" in RandomForestClassifier.
	•	Use SMOTE (Synthetic Minority Over-sampling Technique) to improve positive sentiment detection.

✅ Further Hyperparameter Optimization:
	•	Apply GridSearchCV or RandomizedSearchCV to tune model parameters.
