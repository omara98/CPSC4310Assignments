# News groups dataset evaluated using Naive Bayes classifier
# (Dataset is imported in vectorized form from scikit datasets)
#
# Author: Omar Alsayed (alsayed@uleth.ca)
# April 2022


from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import classification_report

# fetching and storing training data
training = fetch_20newsgroups_vectorized(subset='train')

# fetching and storing test data
testing = fetch_20newsgroups_vectorized(subset='test')

# Naive bayes with laplace smoothing parameter = 1.0
classifier = MultinomialNB()

# Training the classifier
classifier.fit(training.data, training.target)

#prediction
pred = classifier.predict(testing.data)

# Evaluation
print("\nEvaluation (Smoothing parameter = 1.0)\n")

print (classification_report(testing.target, pred))


