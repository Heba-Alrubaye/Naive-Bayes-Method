
import sys
import csv
import numpy as np



def getdata(file):
    array = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            array.append([])  # 2D array
            array[-1] = list(np.fromstring(row[0], dtype=int, sep=' '))
    # print(array)
    return array


# Separate data by class:
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = []
            separated[class_value].append(vector)
        else:
            separated[class_value].append(vector)
    # print(separated)
    return separated


# separated = {0:[], 1:[]}
def table(dataset):
    countUnspam =[1]*12
    countspam = [1]*12
    separated=separateByClass(dataset)
    unspamNo = len(separated[0])+1
    for i in range(len(separated[0])):
        for j in range(12):
            row = separated[0][i]
            if row[j] == 1:
                countUnspam[j]+=1
    proUnspam = np.array(countUnspam)/(unspamNo+1)
    spamNo = len(separated[1])+1
    for i in range(len(separated[1])):
        for j in range(12):
            row = separated[1][i]
            if row[j] == 1:
                countspam[j]+=1
    proSpam = np.array(countspam)/(spamNo+1)
    total =spamNo + unspamNo
    SpamPro = spamNo /total
    UnspamPro = unspamNo /total
    return proSpam , proUnspam , SpamPro , UnspamPro

def predict (dataset, proSpam, SpamPro):
    probality=[1]*len(dataset)
    #P(X = instance|Spam)
    #P(X = instance|NonSpam)
    #Predicted_Class = max(P(X =instance|Spam), P(X instance|NonSpam))
    for i in range(len(dataset)):
        row =dataset[i]
        for j in range(12):
            if row[j]==1:
                probality[i] *=proSpam[j]
            else:
               probality[i] *=(1-proSpam[j])
        probality[i] *= SpamPro
    return probality

def main():
    args = {'file1': sys.argv[1], 'file2': sys.argv[2]}
    trainingSet = getdata(args['file1'])
    testSet = getdata(args['file2'])

    proSpam, proUnspam, SpamPro, UnspamPro = table(trainingSet)
    reslut_spam = predict(testSet,proSpam ,SpamPro)
    reslut_unspam = predict(testSet,proUnspam ,UnspamPro)
    for j in range(len(proSpam)):
        print("P(feature_{}=1|SPAM) = {}".format(j,proSpam[j]))
    for j in range(len(proSpam)):
        print("P(feature_{}=1|NONSPAM) = {}".format(j,proUnspam[j]))
    for j in range(len(reslut_spam)):
        print("P(SPAM|instance_{}) = {}".format(j,reslut_spam[j]))
        print("P(NONSPAM|instance_{}) = {}".format(j, reslut_unspam[j]))

    predictions = []
    for i in range(len(reslut_spam)):
        if reslut_spam[i]>reslut_unspam[i]:
            predictions.append(1)
        else:
            predictions.append(0)
    print(predictions)
    corr = 0
    for j in range(len(testSet)):
        if predictions[j] == testSet[j][-1]:
            corr+=1
        print("Predicted: {} Actual: {}".format(predictions[j], testSet[j][-1]))
    print("Accuracy: {}".format(corr/len(testSet)))
if __name__ == "__main__":
    main()
