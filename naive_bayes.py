
from __future__ import print_function

from pyspark import SparkContext
# $example on$
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint


def parseLine(line):
    parts = line.split(',')
    label = float(parts[len(parts)-1])
    features = Vectors.dense([float(parts[x]) for x in range(0,len(parts)-1)])
    return LabeledPoint(label, features)
# $example off$

if __name__ == "__main__":

    sc = SparkContext(appName="PythonNaiveBayes")

    
    data = sc.textFile('diabetes.csv').map(parseLine)

    # Split data aproximately into training (80%) and test (40%)
    training, test = data.randomSplit([0.8, 0.2], seed=0)

    # Train a naive Bayes model.
    model = NaiveBayes.train(training, 1.0)

    # Make prediction and test accuracy.
    predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda x: x[0] == x[1]).count() / test.count()
    print(accuracy)
    
