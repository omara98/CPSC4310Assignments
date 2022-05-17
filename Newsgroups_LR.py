# News groups dataset evaluated using Logistic regression classifier
# (Dataset is imported in vectorized form from scikit datasets)
#
# Author: Omar Alsayed (alsayed@uleth.ca)
# April 2022


from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report

# fetching and storing training data
training = fetch_20newsgroups_vectorized(subset='train')

# fetching and storing test data
testing = fetch_20newsgroups_vectorized(subset='test')

# Logistic Regression Classifier
classifier = LogisticRegression(max_iter=1000)

# Training the classifier
classifier.fit(training.data, training.target)

#prediction
pred = classifier.predict(testing.data)

# Evaluation
print("\nEvaluation (Logistic Regression)\n")

print (classification_report(testing.target, pred))


