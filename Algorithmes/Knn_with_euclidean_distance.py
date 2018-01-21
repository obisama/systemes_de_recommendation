import csv
import random
import math
import operator


#the function import the datasets from a system file.
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with   open(filename, 'rb')   as   csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

# this function calculate the euclidean distance  between two nodes.
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

#This return the k nearest neighbors based on the euclidean distance 
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][- 1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

# This function evaluate the the result based by  
# calculating the percentage of relevant results on the testSet.
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][- 1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    # Preparing the datasets.
    trainingSet = []
    testSet = []
    split = 0.67 # 67% of the data will be used in the training the rest will be for the test.
    loadDataset('/home/ok/iris.csv', split, trainingSet, testSet)
    print   'Train set: ' + repr(len(trainingSet))
    print   'Test set: ' + repr(len(testSet))
    
    # Getting Predictions.
    predictions = []
    k = 3 # We get the three nearest neighbors
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        # we print the estimated and the real ratings.
        print ('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][- 1]))
    # Finally we calculate the Accuracy porcentage.   
    accuracy = getAccuracy(testSet, predictions)
    print ('Accuracy: ' + repr(accuracy) + '%')


main()