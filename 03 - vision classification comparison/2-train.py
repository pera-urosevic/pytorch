from data import collections
from classifier_simple import SimpleClassifier
from classifier_cnn import CNNClassifier

for collection in collections:
    simple = SimpleClassifier(collection)
    simple.train()

    cnn = CNNClassifier(collection)
    cnn.train()
