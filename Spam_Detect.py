
# Import libraries
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

# Load the data
from google.colab import files  # Use to load data on Google Colab

uploaded = files.upload()  # Use to load data on Google Colab

df = pd.read_csv('emails.csv')  # read the CSV file
df.head(5)

# Print the shape (Get the number of rows and cols)
df.shape

# Get the column names
df.columns

# Checking for duplicates and removing them
df.drop_duplicates(inplace=True)

# Show the new shape (number of rows & columns)
df.shape

# Show the number of missing (NAN, NaN, na) data for each column
df.isnull().sum()

# Need to download stopwords
nltk.download('stopwords')



def process_text(text):
    '''
    What will be covered:
    1. Remove punctuation
    2. Remove stopwords
    3. Return list of clean text words
    '''

    # 1
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    # 2
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    # 3
    return clean_words


# Show the Tokenization
df['text'].head().apply(process_text)


# Print the text (aka the email message)
message4 = 'hello world hello hello world play'  # df['text'][3]
message5 = 'test test test test one hello'
print(message4)
print()

# Convert a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer

bow4 = CountVectorizer(analyzer=process_text).fit_transform([[message4], [message5]])
print(bow4)
print()


from sklearn.feature_extraction.text import CountVectorizer


messages_bow = CountVectorizer(analyzer=process_text).fit_transform(df['text'])


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(messages_bow, df['spam'], test_size=0.20, random_state=0)

# Get the shape of messages_bow
messages_bow.shape


from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Print the predictions
print(classifier.predict(X_train))

# Print the actual values
print(y_train.values)

# Evaluate the model on the training data set
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

pred = classifier.predict(X_train)
print(classification_report(y_train, pred))
print('Confusion Matrix: \n', confusion_matrix(y_train, pred))
print()
print('Accuracy: ', accuracy_score(y_train, pred))

# Print the predictions
print('Predicted value: ', classifier.predict(X_test))

# Print Actual Label
print('Actual value: ', y_test.values)

# Evaluate the model on the test data set
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

pred = classifier.predict(X_test)
print(classification_report(y_test, pred))

print('Confusion Matrix: \n', confusion_matrix(y_test, pred))
print()
print('Accuracy: ', accuracy_score(y_test, pred))