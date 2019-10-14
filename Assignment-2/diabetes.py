import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score
import mlrose
import time
import matplotlib.style as style
import random
import pandas as pd
import numpy as np

def NNetwork1(X, y, X_train, X_test, y_train, y_test):
    # Decision tree code for different degrees of polynomial graph
    mlpTrainingScoreList = []
    mlpTestingScoreList = []

    maxIt = 500
    learningRates = [0.01,0.05, 0.07, 0.1]

    rhcTrainingScoreList = []
    rhcTestingScoreList = []
    for learningRate in learningRates:
        print("Running RHC Learning rate: " + str(learningRate))
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes=None, activation='relu', \
        algorithm='random_hill_climb', max_iters=maxIt, bias=True, is_classifier=True, \
        learning_rate=learningRate, early_stopping=False, clip_max=10000000000.0, restarts=0,\
        pop_size=200, mutation_prob=0.1, max_attempts=10, random_state=None)
        nn_model1.fit(X_train, y_train)
        y_train_pred = nn_model1.predict(X_train)
        y_train_accuracy = accuracy_score(y_train, y_train_pred)
        y_test_pred = nn_model1.predict(X_test)
        y_test_accuracy = accuracy_score(y_test, y_test_pred)

        rhcTrainingScoreList.append(y_train_accuracy)
        rhcTestingScoreList.append(y_test_accuracy)
        print(y_train_accuracy)

    for learningRate in learningRates:
        print("Running MLP Learning rate: " + str(learningRate))
        classifier = MLPClassifier(learning_rate_init=learningRate, max_iter=maxIt)
        classifier.fit(X_train, y_train)
        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)

        mlpTrainingScoreList.append(trainingScore)
        mlpTestingScoreList.append(testingScore)
        print(trainingScore)

    saTrainingScoreList = []
    saTestingScoreList = []
    for learningRate in learningRates:
        print("Running SA Learning rate: " + str(learningRate))
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes=None, activation='relu', \
        algorithm='simulated_annealing', max_iters=maxIt, bias=True, is_classifier=True, \
        learning_rate=learningRate, early_stopping=False, clip_max=10000000000.0, restarts=0,\
        pop_size=200, mutation_prob=0.1, max_attempts=10, random_state=None)
        nn_model1.fit(X_train, y_train)
        y_train_pred = nn_model1.predict(X_train)
        y_train_accuracy = accuracy_score(y_train, y_train_pred)
        y_test_pred = nn_model1.predict(X_test)
        y_test_accuracy = accuracy_score(y_test, y_test_pred)

        saTrainingScoreList.append(y_train_accuracy)
        saTestingScoreList.append(y_test_accuracy)
        print(y_train_accuracy)

    gaTrainingScoreList = []
    gaTestingScoreList = []
    for learningRate in learningRates:
        print("Running GA Learning rate: " + str(learningRate))
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes=None, activation='relu', \
        algorithm='genetic_alg', max_iters=maxIt, bias=True, is_classifier=True, \
        learning_rate=learningRate, early_stopping=False, clip_max=10000000000.0, restarts=0,\
        pop_size=200, mutation_prob=0.1, max_attempts=10, random_state=None)
        nn_model1.fit(X_train, y_train)
        y_train_pred = nn_model1.predict(X_train)
        y_train_accuracy = accuracy_score(y_train, y_train_pred)
        y_test_pred = nn_model1.predict(X_test)
        y_test_accuracy = accuracy_score(y_test, y_test_pred)

        gaTrainingScoreList.append(y_train_accuracy)
        gaTestingScoreList.append(y_test_accuracy)
        print(y_train_accuracy)

    gdTrainingScoreList = []
    gdTestingScoreList = []
    for learningRate in learningRates:
        print("Running GD Learning rate: " + str(learningRate))
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes=None, activation='relu', \
        algorithm='gradient_descent', max_iters=maxIt, bias=True, is_classifier=True, \
        learning_rate=learningRate, early_stopping=False, clip_max=10000000000.0, restarts=0,\
        pop_size=200, mutation_prob=0.1, max_attempts=10, random_state=None)
        nn_model1.fit(X_train, y_train)
        y_train_pred = nn_model1.predict(X_train)
        y_train_accuracy = accuracy_score(y_train, y_train_pred)
        y_test_pred = nn_model1.predict(X_test)
        y_test_accuracy = accuracy_score(y_test, y_test_pred)

        gdTrainingScoreList.append(y_train_accuracy)
        gdTestingScoreList.append(y_test_accuracy)
        print(y_train_accuracy)


    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    titleStr = "Learning Rate vs Training Set Score"
    plt.title(titleStr)
    plt.xlabel("Learning Rate")
    plt.ylabel("Training Set Accuracy")
    plt.plot(learningRates, mlpTrainingScoreList, '-o', label="MLP Classifier Training Accuracy")
    plt.plot(learningRates, rhcTrainingScoreList, '-o', label="RHC NN Training Accuracy")
    plt.plot(learningRates, saTrainingScoreList, '-o', label="SA NN Training Accuracy")
    plt.plot(learningRates, gaTrainingScoreList, '-o', label="GA NN Training Accuracy")
    plt.plot(learningRates, gdTrainingScoreList, '-o', label="Gradient Descent NN Training Accuracy")
    plt.legend(loc='best')
    saveFigPath = "figures/Learning-RatevsTraining-SetScore.png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    titleStr = "Learning Rate vs Testing Set Score"
    plt.title(titleStr)
    plt.xlabel("Learning Rate")
    plt.ylabel("Testing Set Accuracy")
    plt.plot(learningRates, mlpTestingScoreList, '-o', label="MLP Classifier Testing Accuracy")
    plt.plot(learningRates, rhcTestingScoreList, '-o', label="RHC NN Testing Accuracy")
    plt.plot(learningRates, saTestingScoreList, '-o', label="SA NN Testing Accuracy")
    plt.plot(learningRates, gaTestingScoreList, '-o', label="GA NN Testing Accuracy")
    plt.plot(learningRates, gdTestingScoreList, '-o', label="Gradient Descent NN Training Accuracy")
    plt.legend(loc='best')
    saveFigPath = "figures/Learning-RatevsTesting-SetScore.png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    mlpTrainingScoreList.clear()
    mlpTestingScoreList.clear()
    rhcTrainingScoreList.clear()
    rhcTestingScoreList.clear()
    saTrainingScoreList.clear()
    saTestingScoreList.clear()
    gaTrainingScoreList.clear()
    gaTestingScoreList.clear()
    gdTrainingScoreList.clear()
    gdTestingScoreList.clear()

    mlpRunTime = []
    maxIters = [500, 1000, 1500, 2000]
    for maxIter in maxIters:
        print("Running MLP MAX iter: " + str(maxIter))
        classifier = MLPClassifier(learning_rate_init=0.0003, max_iter=maxIter)
        startTime = time.time()
        classifier.fit(X_train, y_train)
        endTime = time.time()
        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)

        mlpTrainingScoreList.append(trainingScore)
        mlpTestingScoreList.append(testingScore)
        mlpRunTime.append(endTime-startTime)
        print(trainingScore)

    rhcRunTime = []
    for maxIter in maxIters:
        print("Running RHC MAX iter: " + str(maxIter))
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes=None, activation='relu', \
        algorithm='random_hill_climb', max_iters=maxIter, bias=True, is_classifier=True, \
        learning_rate=0.09, early_stopping=False, clip_max=10000000000.0, restarts=0,\
        pop_size=200, mutation_prob=0.1, max_attempts=10, random_state=None)
        startTime = time.time()
        nn_model1.fit(X_train, y_train)
        endTime = time.time()
        y_train_pred = nn_model1.predict(X_train)
        y_train_accuracy = accuracy_score(y_train, y_train_pred)
        y_test_pred = nn_model1.predict(X_test)
        y_test_accuracy = accuracy_score(y_test, y_test_pred)

        rhcTrainingScoreList.append(y_train_accuracy)
        rhcTestingScoreList.append(y_test_accuracy)
        rhcRunTime.append(endTime-startTime)
        print(y_train_accuracy)

    saRunTime = []
    for maxIter in maxIters:
        print("Running SA MAX iter: " + str(maxIter))
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes=None, activation='relu', \
        algorithm='simulated_annealing', max_iters=maxIter, bias=True, is_classifier=True, \
        learning_rate=1, early_stopping=False, clip_max=10000000000.0, restarts=0,\
        pop_size=200, mutation_prob=0.1, max_attempts=10, random_state=None)
        startTime = time.time()
        nn_model1.fit(X_train, y_train)
        endTime = time.time()
        y_train_pred = nn_model1.predict(X_train)
        y_train_accuracy = accuracy_score(y_train, y_train_pred)
        y_test_pred = nn_model1.predict(X_test)
        y_test_accuracy = accuracy_score(y_test, y_test_pred)

        saTrainingScoreList.append(y_train_accuracy)
        saTestingScoreList.append(y_test_accuracy)
        saRunTime.append(endTime-startTime)
        print(y_train_accuracy)

    gaRunTime = []
    for maxIter in maxIters:
        print("Running GA MAX iter: " + str(maxIter))
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes=None, activation='relu', \
        algorithm='genetic_alg', max_iters=maxIter, bias=True, is_classifier=True, \
        learning_rate=0.02, early_stopping=False, clip_max=10000000000.0, restarts=0,\
        pop_size=200, mutation_prob=0.1, max_attempts=10, random_state=None)
        startTime = time.time()
        nn_model1.fit(X_train, y_train)
        endTime = time.time()
        y_train_pred = nn_model1.predict(X_train)
        y_train_accuracy = accuracy_score(y_train, y_train_pred)
        y_test_pred = nn_model1.predict(X_test)
        y_test_accuracy = accuracy_score(y_test, y_test_pred)

        gaTrainingScoreList.append(y_train_accuracy)
        gaTestingScoreList.append(y_test_accuracy)
        gaRunTime.append(endTime-startTime)
        print(y_train_accuracy)

    gdRunTime = []
    for maxIter in maxIters:
        print("Running GD MAX iter: " + str(maxIter))
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes=None, activation='relu', \
        algorithm='gradient_descent', max_iters=maxIter, bias=True, is_classifier=True, \
        learning_rate=0.03, early_stopping=False, clip_max=10000000000.0, restarts=0,\
        pop_size=200, mutation_prob=0.1, max_attempts=10, random_state=None)
        startTime = time.time()
        nn_model1.fit(X_train, y_train)
        endTime = time.time()
        y_train_pred = nn_model1.predict(X_train)
        y_train_accuracy = accuracy_score(y_train, y_train_pred)
        y_test_pred = nn_model1.predict(X_test)
        y_test_accuracy = accuracy_score(y_test, y_test_pred)

        gdTrainingScoreList.append(y_train_accuracy)
        gdTestingScoreList.append(y_test_accuracy)
        gdRunTime.append(endTime-startTime)
        print(y_train_accuracy)

    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    titleStr = "Max Iteration vs Training Set Score"
    plt.title(titleStr)
    plt.xlabel("Max Iterations")
    plt.ylabel("Training Set Accuracy")
    plt.plot(maxIters, mlpTrainingScoreList, '-o', label="MLP Classifier Training Accuracy")
    plt.plot(maxIters, rhcTrainingScoreList, '-o', label="RHC NN Training Accuracy")
    plt.plot(maxIters, saTrainingScoreList, '-o', label="SA NN Training Accuracy")
    plt.plot(maxIters, gaTrainingScoreList, '-o', label="GA NN Training Accuracy")
    plt.plot(maxIters, gdTrainingScoreList, '-o', label="Gradient Descent NN Training Accuracy")
    plt.legend(loc='best')
    saveFigPath = "figures/Max-IterationsvsTraining-SetScore.png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    titleStr = "Max Iteration vs Testing Set Score"
    plt.title(titleStr)
    plt.xlabel("Max Iterations")
    plt.ylabel("Testing Set Accuracy")
    plt.plot(maxIters, mlpTestingScoreList, '-o', label="MLP Classifier Testing Accuracy")
    plt.plot(maxIters, rhcTestingScoreList, '-o', label="RHC NN Testing Accuracy")
    plt.plot(maxIters, saTestingScoreList, '-o', label="SA NN Testing Accuracy")
    plt.plot(maxIters, gaTestingScoreList, '-o', label="GA NN Testing Accuracy")
    plt.plot(maxIters, gdTestingScoreList, '-o', label="Gradient Descent NN Testing Accuracy")
    plt.legend(loc='best')
    saveFigPath = "figures/Max-IterstionsvsTesting-SetScore.png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    titleStr = "Max Iteration vs Run Time"
    plt.title(titleStr)
    plt.xlabel("Max Iterations")
    plt.ylabel("Run Time")
    plt.plot(maxIters, mlpRunTime, '-o', label="MLP Run Time")
    plt.plot(maxIters, rhcRunTime, '-o', label="RHC Run Time")
    plt.plot(maxIters, saRunTime, '-o', label="SA Run Time")
    plt.plot(maxIters, gaRunTime, '-o', label="GA Run Time")
    plt.plot(maxIters, gdRunTime, '-o', label="Gradient Descent Run Time")
    plt.legend(loc='best')
    saveFigPath = "figures/Max-IterstionsvsRun-Time.png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()
# def plotModelLearningCurve(classifierName, variableName, variableList, trainingScoreList, testingScoreList, crossValScoreList, dataset):
#     plt.style.use('seaborn-poster') #sets the size of the charts
#     plt.style.use('ggplot')
#     titleStr = "Learning Curve - " + str(classifierName)
#     plt.title(titleStr)
#     plt.xlabel(variableName)
#     plt.ylabel("Score")
#     plt.plot(variableList, trainingScoreList, '-o', color = "red", label="Training Set Score")
#     plt.plot(variableList, testingScoreList, '-o', color = "blue", label="Testing Set Score")
#     plt.plot(variableList, crossValScoreList, '-o', color = "green", label="CV Testing Set Score")
#     plt.legend(loc='best')
#     saveFigPath = "figuresDiabetes/" + dataset + titleStr + ".png"
#     plt.savefig(saveFigPath)
#     plt.clf()
#     plt.cla()
#     plt.close()

def rocCurve1(X, y, X_train, X_test, y_train, y_test):
    maxIt = 500
    lw = 2
    def calcPlot(classifier, color, clfName):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()


        # Compute micro-average ROC curve and ROC area
        classifier.fit(X_train, y_train)
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test, classifier.predict(X_test))
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.plot(fpr["micro"], tpr["micro"], color=color,
                lw=lw, label=clfName + (' (area = %0.2f)' % roc_auc["micro"]))
    # SVM
    plt.figure()
    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes=None, activation='relu', \
    algorithm='random_hill_climb', max_iters=maxIt, bias=True, is_classifier=True, \
    learning_rate=0.09, early_stopping=False, clip_max=10000000000.0, restarts=0,\
    pop_size=200, mutation_prob=0.1, max_attempts=10, random_state=None)
    nn_model1.fit(X_train, y_train)
    calcPlot(nn_model1, "deeppink", "RHC")
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes=None, activation='relu', \
    algorithm='simulated_annealing', max_iters=maxIt, bias=True, is_classifier=True, \
    learning_rate=1, early_stopping=False, clip_max=10000000000.0, restarts=0,\
    pop_size=200, mutation_prob=0.1, max_attempts=10, random_state=None)
    nn_model1.fit(X_train, y_train)
    calcPlot(nn_model1, "navy", "SA")
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes=None, activation='relu', \
    algorithm='genetic_alg', max_iters=maxIt, bias=True, is_classifier=True, \
    learning_rate=0.02, early_stopping=False, clip_max=10000000000.0, restarts=0,\
    pop_size=200, mutation_prob=0.1, max_attempts=10, random_state=None)
    nn_model1.fit(X_train, y_train)
    calcPlot(nn_model1, "aqua", "GA")
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes=None, activation='relu', \
    algorithm='gradient_descent', max_iters=maxIt, bias=True, is_classifier=True, \
    learning_rate=0.03, early_stopping=False, clip_max=10000000000.0, restarts=0,\
    pop_size=200, mutation_prob=0.1, max_attempts=10, random_state=None)
    nn_model1.fit(X_train, y_train)
    calcPlot(nn_model1, "darkorange", "Gradient Descent")
    classifier = MLPClassifier(learning_rate_init=0.0004)
    classifier.fit(X_train, y_train)
    calcPlot(classifier, "cornflowerblue", "MLP")
    plt.plot([0, 1], [0, 1], lw=lw, color='black', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    saveFigPath = "figures/ROC-curve.png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

def main():
    diabetesData = pd.read_csv("diabetes.csv")
    diabetesData = pd.DataFrame(diabetesData.values, columns=list(diabetesData))

    X = diabetesData.values[:,0:8]
    y = diabetesData.values[:,8]

    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    
    NNetwork1(X, y, X_train, X_test, y_train, y_test)
    rocCurve1(X, y, X_train, X_test, y_train, y_test)

    # X = pd.read_csv("mnist_train.csv").values
    # y = pd.read_csv("mnist_test.csv").values

    # # X_train = X[0:60000,1:]
    # # y_train = X[0:60000,0]

    # X_train = X[0:5000,1:]
    # y_train = X[0:5000,0]

    # y_train = y_train == 7

    # # X_test = y[:10000, 1:]
    # # y_test = y[:10000, 0]
    # X_test = y[:1000, 1:]
    # y_test = y[:1000, 0]

    # y_test = y_test == 7

    # # print(y_train)

    # # SVM2(X, y, X_train, X_test, y_train, y_test)
    # # DT2(X, y, X_train, X_test, y_train, y_test)
    # # KNN2(X, y, X_train, X_test, y_train, y_test)
    # # ADABoost2(X, y, X_train, X_test, y_train, y_test)
    # NNetwork1(X, y, X_train, X_test, y_train, y_test)
    # # rocCurve2(X, y, X_train, X_test, y_train, y_test)

if __name__== "__main__":
    main()
