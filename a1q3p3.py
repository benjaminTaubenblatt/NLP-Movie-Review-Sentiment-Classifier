# Benjamin Taubenblatt
# COMP 550 Assignment 1
# Python 3

import random

from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import sklearn.metrics
from sklearn .feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier

# load our data
all_reviews = []

def load_data(file_path):
    # Positive reviews
    with open(file_path + '.pos','r', encoding='latin-1') as f:
        pos_data = f.readlines()
        for line in pos_data:
            all_reviews.append((line, 'positive'))

    # Negative reviews
    with open(file_path + '.neg','r', encoding='latin-1') as g:
        neg_data = g.readlines()
        for line in neg_data:
            all_reviews.append((line, 'negative'))

file_path = './rt-polaritydata/rt-polarity'
load_data(file_path)

# clean the data
def clean_data(input, new_stopwords):
    lemmatizer = WordNetLemmatizer()
    # NLTK stopwords
    nltk_stopwords = stopwords.words('english')
    # Total stopwords
    total_stopwords = nltk_stopwords + new_stopwords
    cleaned_data = ''
    # Tokenize and strip words
    all_words = word_tokenize(input)

    for word in all_words:
        current_word = lemmatizer.lemmatize(word.lower())
        if current_word not in total_stopwords and len(current_word) > 2:
            cleaned_data += current_word + ' '
    return cleaned_data.rstrip()


# clean and organize data
all_clean_reviews_and_sentiments = []
for (review, sentiment) in all_reviews:
    clean_review = clean_data(review, [])
    all_clean_reviews_and_sentiments.append((clean_review, sentiment))

# Note: the cleaned data (no stopwords) is not used as it performed worse than with stopwords
#random.shuffle(all_clean_reviews_and_sentiments)

# shuffle data
random.shuffle(all_reviews)


# Split the 2-tuples into reviews and sentiments
reviews = []
sentiments = []

for (review, sentiment) in all_reviews:
    reviews.append(review)
    sentiments.append(sentiment)
'''
for (review, sentiment) in all_clean_reviews_and_sentiments:
    reviews.append(review)
    sentiments.append(sentiment) '''

# Split data set into train and test
x_train, x_test, y_train, y_test = train_test_split(reviews, sentiments, test_size=0.30, random_state=42)


# ----- Unigram -----
unigram_vectorizer = CountVectorizer(ngram_range = (1,1), min_df = 1)
# Fit to training data
X_unigram = unigram_vectorizer.fit_transform(x_train)

# ----- Bigram -----
bigram_vectorizer = CountVectorizer(ngram_range=(2,2), min_df = 1)
# Fit to training data
X_bigram = bigram_vectorizer.fit_transform(x_train)

# ----- Unigram and Bigram -----
unigram_and_bigram_vectorizer = CountVectorizer(ngram_range=(1,2), min_df = 1)
# Fit to training data
X_both = unigram_and_bigram_vectorizer.fit_transform(x_train)

def run_algorithm(name, classifier):
    print("----- " + name + " -----\n")

    # ----- Unigram -----
    # Train classifier
    classifier.fit(X_unigram, y_train)

    # Test model:
    X_test = unigram_vectorizer.transform(x_test)
    y_pred = classifier.predict(X_test)

    # Accuracy
    print("Unigram Accuracy: ", sklearn.metrics.accuracy_score(y_test, y_pred))
    print("Confusion Matrix: \n")
    print(sklearn.metrics.confusion_matrix(y_test, y_pred, labels=['positive', 'negative']))
    print('\n')

    # ----- Bigram ------
    # Train classifier
    classifier.fit(X_bigram, y_train)

    # Test model:
    X_test = bigram_vectorizer.transform(x_test)
    y_pred = classifier.predict(X_test)

    # Accuracy
    print("Bigram Accuracy: ", sklearn.metrics.accuracy_score(y_test, y_pred))
    print("Confusion Matrix: \n")
    print(sklearn.metrics.confusion_matrix(y_test, y_pred, labels=['positive', 'negative']))
    print('\n')

    # ----- Unigram and Bigram -----
    # Train classifier
    classifier.fit(X_both, y_train)

    # Test model:
    X_test = unigram_and_bigram_vectorizer.transform(x_test)
    y_pred = classifier.predict(X_test)

    # Accuracy
    print("Bigram and Unigram Accuracy: ", sklearn.metrics.accuracy_score(y_test, y_pred))
    print("Confusion Matrix: \n")
    print(sklearn.metrics.confusion_matrix(y_test, y_pred, labels=['positive', 'negative']))
    print('\n')

# Support Vector Machine w/ Linear Kernel
svm_classifier = LinearSVC()
run_algorithm("Support Vector Machine w/ Linear Kernel", svm_classifier)

# Logistic Regression
logistic_regression_classifier = LogisticRegression()
run_algorithm("Logistic Regression", logistic_regression_classifier)

# Naive Bayes
nb_classifier = MultinomialNB()
run_algorithm("Naive Bayes", nb_classifier)

# DummyClassifier
dummy_classifier = DummyClassifier()
run_algorithm("Dummy Classifier", dummy_classifier)
