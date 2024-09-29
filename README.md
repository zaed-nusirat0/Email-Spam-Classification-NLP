# Spam Email Classification using Natural Language Processing (NLP)
Project Overview
This project aims to develop a robust spam email classification system leveraging Natural Language Processing (NLP) and various machine learning algorithms. As spam emails continue to pose challenges in email management, this project seeks to provide an effective solution to classify emails into two categories: 'spam' and 'ham' (non-spam).

Objectives
To preprocess and clean email data for effective model training.
To visualize the distribution and characteristics of the dataset.
To implement and evaluate multiple machine learning models for spam detection.
To achieve high accuracy in classification, with the final model reaching a testing accuracy of 98%.
Dataset
The dataset used in this project is sourced from Kaggle, comprising a collection of emails labeled as spam or ham. The dataset provides a realistic scenario for testing the classification models.

Key Features
Data Preprocessing:

Utilizes NLTK and BeautifulSoup to clean and preprocess the email text, removing HTML tags and irrelevant characters.
Tokenization, stopword removal, and lemmatization are applied to prepare the text for analysis.
Data Visualization:

Employs libraries like Matplotlib and Seaborn to visualize the data, offering insights into the distribution of spam and ham emails.
Generates word clouds to highlight frequently occurring words in spam and ham emails.
Machine Learning Models:

Implements various classification algorithms including:
Logistic Regression
Random Forest Classifier
Support Vector Classifier (SVC)
K-Nearest Neighbors (KNN)
Naive Bayes
Each model is trained and evaluated based on metrics such as accuracy, precision, recall, and F1-score.
Model Evaluation:

Utilizes confusion matrices and ROC curves to assess the performance of each model.
The Random Forest Classifier achieves a training accuracy of 100% and a testing accuracy of 97.49%, indicating effective learning with minimal overfitting.
User-Friendly Prediction Interface:

A simple function to allow users to input a message and receive predictions on whether it is spam or ham.
