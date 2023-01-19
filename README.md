<h1>Simple Email Spam Detector</h1>

<h2>Description</h2>
Project consists of a simple PowerShell script that walks the user through "zeroing out" (wiping) any drives that are connected to the system. The utility allows you to select the target disk and choose the number of passes that are performed. The PowerShell script will configure a diskpart script file based on the user's selections and then launch Diskpart to perform the disk sanitization.
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




