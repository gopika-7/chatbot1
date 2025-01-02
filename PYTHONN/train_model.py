# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 09:04:50 2025

@author: Gopika
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Example dataset
X_data = [
    "Hello", "Hi there", "Good morning", "How are you?",
    "What's your age?", "How old are you?", "What is your name?",
    "Goodbye", "Bye", "See you later"
]
y_data = [
    "greeting", "greeting", "greeting", "greeting",
    "age", "age", "name",
    "farewell", "farewell", "farewell"
]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Create and train the vectorizer
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)

# Create and train the classifier
clf = RandomForestClassifier()
clf.fit(X_train_vect, y_train)

# Save the trained vectorizer and classifier
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Vectorizer saved as 'vectorizer.pkl'.")

joblib.dump(clf, 'classifier_model.pkl')
print("Classifier saved as 'classifier_model.pkl'.")
