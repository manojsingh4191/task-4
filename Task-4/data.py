# 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load the dataset
# Here we use a common dataset such as the 'SpamAssassin' dataset or an existing CSV dataset for email classification
# You can also use the sklearn's "load_files" method for text classification

# Let's assume you have a CSV file with 'text' and 'label' columns
# where label 1 = 'spam' and label 0 = 'ham' (non-spam)

# Load the dataset into pandas
data = pd.read_csv('spam_emails.csv')

# Display the first few rows
print(data.head())

# 3. Preprocess the data

# Extracting features (email content) and labels (spam or not spam)
X = data['text']  # Feature: Email content
y = data['label']  # Target: Spam (1) or Ham (0)

# Convert text data into numerical format using CountVectorizer (Bag-of-Words model)
vectorizer = CountVectorizer(stop_words='english')  # Remove common stopwords
X_transformed = vectorizer.fit_transform(X)

# 4. Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# 5. Train the Model using Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Make predictions on the test set
y_pred = model.predict(X_test)

# 7. Evaluate the Model
# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion matrix to visualize True Positives, False Positives, True Negatives, and False Negatives
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Classification Report (precision, recall, f1-score)
report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)

# 8. (Optional) Hyperparameter Tuning or Model Enhancement (e.g., trying different algorithms)
# You can try other models such as Logistic Regression, Support Vector Machine, etc.
