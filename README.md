<h1>Simple Email Spam Detector</h1>

<h2>Description</h2>
This is a Python script that uses the pandas, nltk and sklearn library to create a simple Email Spam Detector. The script starts by installing the necessary libraries and modules, then it imports the necessary libraries. After that the script reads the emails dataset, tokenize the email text, create a bag of words representation, split the data into training and testing sets. It trains a Naive Bayes classifier on the training data and tests it on the testing data. Finally, it evaluates the performance of the classifier using metrics such as accuracy, precision, recall and f1-score, and print the results.
<br />


<h2>Languages and Utilities Used</h2>

- <b>Python</b> 


<h2>Environments Used </h2>

- <b>Windows 11</b>
- <b>Jupyter Notebook</b>

<h2>Program code walk-through:</h2>

<b>Step 1: Install necessary libraries and modules:</b>
```python
pip install pandas
pip install nltk
pip install sklearn
```
<b>Step 1: Install necessary libraries and modules:</b>
```python
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
```
<b>Step 3: Load the dataset of emails, where the dataset is labeled as spam or ham:</b>
```python
data = pd.read_csv('emails.csv')
```
<b>Step 4: Prepare the data for modeling. This includes tokenizing the email text and creating a bag of words representation:</b>
```python 
# Tokenize the email text
data['email'] = data['email'].apply(word_tokenize)

# Create bag of words representation
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['email'])
```
<b>Step 5: Split the data into training and testing sets:</b>
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, data['label'], test_size=0.2)
```
<b>Step 6: Train a Naive Bayes classifier on the training data:</b>
```python 
clf = MultinomialNB()
clf.fit(X_train, y_train)
```
<b>Step 7: Test the classifier on the testing data:</b>
```python
y_pred = clf.predict(X_test)
```
<b>Step 8: Evaluate the performance of the classifier using metrics such as accuracy, precision, recall and f1-score</b>
```python 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label='spam')
rec = recall_score(y_test, y_pred, pos_label='spam')
f1 = f1_score(y_test, y_pred, pos_label='spam')

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
```




