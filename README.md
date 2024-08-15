# Fake News Detection using SVM and TF-IDF

This project implements a Fake News Detection system using a Support Vector Machine (SVM) classifier with TF-IDF vectorization. The model is trained to differentiate between real and fake news articles. The dataset used in this project is `fake_or_real_news.csv`.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to build a machine learning model capable of detecting fake news articles. We use TF-IDF vectorization to transform the text data into numerical features, followed by a Linear Support Vector Classifier (LinearSVC) to perform the classification.

## Dataset

The dataset used in this project is the `fake_or_real_news.csv` file. It contains the following columns:
- `text`: The full text of the news article.
- `label`: Indicates whether the news article is "REAL" or "FAKE".

## Dependencies

The following Python libraries are required to run the project:
- `numpy`
- `pandas`
- `scikit-learn`

You can install the dependencies using the following command:

```bash
pip install numpy pandas scikit-learn
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your_username/fake-news-detection.git
cd fake-news-detection
```

2. Install the required Python libraries:

```bash
pip install -r requirements.txt
```

3. Place the dataset file (`fake_or_real_news.csv`) in the root directory.

## Model Training

The model is trained using the following steps:

1. **Preprocessing**: The `label` column is converted into binary values, where "REAL" is mapped to 0 and "FAKE" is mapped to 1.
2. **Train-Test Split**: The dataset is split into training (80%) and testing (20%) subsets.
3. **TF-IDF Vectorization**: The text data is vectorized using TF-IDF (Term Frequency-Inverse Document Frequency) to convert it into numerical features.
4. **Model Training**: A Linear Support Vector Classifier (LinearSVC) is trained on the vectorized text data.

## Evaluation

After training the model, we evaluate its performance using the test set. The model achieves an accuracy of approximately **94%** on the test set.

```python
accuracy = clf.score(x_test_vectorized, Y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
```

## Usage

To predict whether a given news article is fake or real, you can use the trained model as follows:

1. Save a custom news article to a text file (`test.txt`).
2. Load the text from the file and vectorize it.
3. Use the trained model to make a prediction.

```python
with open("test.txt", "w", encoding="utf-8") as f: 
    f.write(x_test.iloc[10])  # Example text

with open("test.txt", "r", encoding="utf-8") as f:
    text = f.read()

vectorized_text = vectorizer.transform([text])
prediction = clf.predict(vectorized_text)

if prediction[0] == 0:
    print("The news article is REAL.")
else:
    print("The news article is FAKE.")
```

## Future Work

- Experiment with different machine learning models such as Random Forests, Naive Bayes, or Neural Networks to compare performance.
- Implement additional text preprocessing steps, such as stemming, lemmatization, or n-grams, to improve accuracy.
- Build a user-friendly web application for real-time fake news detection.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
