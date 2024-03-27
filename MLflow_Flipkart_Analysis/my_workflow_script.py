import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

from sklearn.preprocessing import MinMaxScaler

import os
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

from sklearn import metrics

@task
def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)


@task
def split_inputs_output(data, inputs, output):
    """
    Split features and target variables.
    """
    X = data[inputs]
    y = data[output]
    return X, y

@task
def split_train_test(X, y, test_size=0.25, random_state=0):
    """
    Split data into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

@task
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
    
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    
    text = re.sub(r'^b\s+', '', text)
    
    text = text.lower()
    
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

@task 
def lemmatize_text(text):
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    text = ' '.join(text)
    return text

@task

def initialize_vectorizer():
    vectorizer = CountVectorizer()
    return vectorizer

def fit_transform( X_train):
    return self.vectorizer.fit_transform(X_train)

def transform( X_test):
    return self.vectorizer.transform(X_test)
    
@task
def train_model(X_train, y_train, hyperparameters):
    """
    Training the machine learning model.
    """

    nb = MultinomialNB(**hyperparameters)

    nb.fit(X_train, y_train)
    return nb

@task
def evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Evaluating the model.
    """
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_score = metrics.accuracy_score(y_train, y_train_pred)
    test_score = metrics.accuracy_score(y_test, y_test_pred)
    
    return train_score, test_score


# Workflow

@flow(name="Multinomial Naive Baye's Training Flow")

def workflow():
    DATA_PATH = "./final_data.csv"
    INPUTS = "Review text"
    OUTPUT = 'Sentiment_num'
    HYPERPARAMETERS =         {
                'alpha': 1.0,
                'fit_prior': True,
                'class_prior': None
        }
    
    data = load_data(DATA_PATH)

    X, y = split_inputs_output(data, INPUTS, OUTPUT)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    X_train= X_train.apply(clean_text)
    X_train = X_train.apply(lemmatize_text)
    
    X_test=X_test.apply(clean_text)
    X_test=X_test.apply(lemmatize_text)

    
    vectorizer = initialize_vectorizer()
    X_train_bow = fit_transform(vectorizer, X_train)
    X_test_bow = transform(vectorizer, X_test)

    model = train_model(X_train_bow, y_train, HYPERPARAMETERS)
    
    train_score, test_score = evaluate_model(model, X_train_bow, y_train, X_test_bow, y_test)
    
    print("Train Score:", train_score)
    print("Test Score:", test_score)
    
    
if __name__ == "__main__":
    workflow.serve(
        name="my-first-deployment",
        cron="59 2 26 03 2"
    )