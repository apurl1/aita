import pandas as pd
import numpy as np
import ast 
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
import sys

np.set_printoptions(threshold=sys.maxsize)
data = pd.read_csv('processed_data.csv')

dataArr = []
labelsArr = []

for index, row in data.iterrows():
    element = []
    element.extend([float(i) for i in row['Title'][1:-1].split()])
    element.extend([float(i) for i in row['Text'][1:-1].split()])
    element.append(row['Score'])
    element.append(row['Upvote Ratio'])
    element.append(row['Num Awards'])
    element.append(row['Num Comments'])
    element.append(row['Num Crossposts'])
    #element.append(row['Author Flair'])
    senti = ast.literal_eval(row['Polite Score'])
    element.extend([senti['neg'], senti['neu'], senti['pos'], senti['compound']])
    element.append(row['Num Errors'])
    element.append(row['Post Length'])
    dataArr.append(element)
    labelsArr.append(row['Label'])

numPosts = len(dataArr)
trainData = dataArr[:int(numPosts/3*2)]
testData = dataArr[int(numPosts/3*2):]
trainLabels = labelsArr[:int(numPosts/3*2)]
testLabels = labelsArr[int(numPosts/3*2):]

kf = KFold(n_splits=5)

CValues = [0.5, 1, 2]
classWeightValues = [None, 'balanced']

best = 0
bestC = 0
bestWeight = None

for c in CValues:
    for classWeight in classWeightValues:
        svc = SVC(C=c, class_weight=classWeight)
        acc = cross_val_score(svc, trainData, trainLabels, cv=kf).mean()
        if acc > best:
            best = acc
            bestC = c
            bestWeight = classWeight

svc = SVC(C=bestC, class_weight=bestWeight)
svc.fit(trainData, trainLabels)

predictions = svc.predict(testData)
numCorrect = 0
for i in range(len(predictions)):
    if predictions[i] == labelsArr[i]:
        numCorrect += 1
print('Num Correct:', numCorrect)
print('Accuracy:', numCorrect/len(predictions))
print('Num Total:', len(predictions))
print('Best C:', bestC)
print('Best Weights:', bestWeight)
print('Vall Accuracy:', best)