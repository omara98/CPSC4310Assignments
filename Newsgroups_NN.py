# News groups dataset evaluated using Neural Network classifier
# (Dataset is imported in vectorized form from scikit datasets)
#
# Author: Omar Alsayed (alsayed@uleth.ca)
# April 2022


from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
import time

# fetching and storing training data
training = fetch_20newsgroups_vectorized(subset='train')

# fetching and storing test data
testing = fetch_20newsgroups_vectorized(subset='test')

# Neural Network with three hidden layers with 10 units each
classifier = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)

# Training the classifier
classifier.fit(training.data, training.target)

#prediction
pred = classifier.predict(testing.data)

# Evaluation
print("\nEvaluation (Neural Network)\n")

print (classification_report(testing.target, pred))
