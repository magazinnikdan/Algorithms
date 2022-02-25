import math
import matplotlib.pyplot as plt
import numpy as np
import time

# create a plot
fig = plt.figure(1, figsize=[10, 10], dpi=90)
axScatter = plt.subplot(111)


def candMedian(dataPoints):
    # Calculate the first candidate median as the geometric mean
    tempX = 0.0
    tempY = 0.0

    for i in range(0, len(dataPoints)):
        tempX += dataPoints[i][0]
        tempY += dataPoints[i][1]

    return [tempX / len(dataPoints), tempY / len(dataPoints)]


def numersum(testMedian, dataPoint):
    # Provides the denominator of the weiszfeld algorithm depending on whether you are adjusting the candidate x or y
    return 1 / math.sqrt((testMedian[0] - dataPoint[0]) ** 2 + (testMedian[1] - dataPoint[1]) ** 2)


def denomsum(testMedian, dataPoints):
    # Provides the denominator of the weiszfeld algorithm
    temp = 0.0
    for i in range(0, len(dataPoints)):
        temp += 1 / math.sqrt((testMedian[0] - dataPoints[i][0]) ** 2 + (testMedian[1] - dataPoints[i][1]) ** 2)
    return temp


def objfunc(testMedian, dataPoints):
    # This function calculates the sum of linear distances from the current candidate median to all points
    # in the data set, as such it is the objective function we are minimising.
    temp = 0.0
    for i in range(0, len(dataPoints)):
        temp += math.sqrt((testMedian[0] - dataPoints[i][0]) ** 2 + (testMedian[1] - dataPoints[i][1]) ** 2)
    return temp


# Use the above functions to calculate the median
# Test Data - later to be read from a fil

# numIter depends on how long it take to get a suitable convergence of objFunc

def weisz_calc(A, numIter):
    x_step = []
    testMedian = candMedian(A)
    # minimise the objective function.
    for x in range(0, numIter):
        objfunc(testMedian, A)
        denom = denomsum(testMedian, A)
        nextx = 0.0
        nexty = 0.0

        for y in range(0, len(A)):
            nextx += (A[y][0] * numersum(testMedian, A[y])) / denom
            nexty += (A[y][1] * numersum(testMedian, A[y])) / denom
        x_step.append([nextx,nexty])
        testMedian = [nextx, nexty]
    return testMedian, x_step

