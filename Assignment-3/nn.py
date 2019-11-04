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
from sklearn.decomposition import FastICA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import SparseRandomProjection
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

def NNetwork1(X, y, X_train, X_test, y_train, y_test, dataset):
    # Decision tree code for different degrees of polynomial graph
    mlpTrainingScoreList = []
    mlpTestingScoreList = []

    maxIt = 500
    learningRates = [0.01,0.05, 0.07, 0.1]

    for learningRate in learningRates:
        print("Running MLP Learning rate: " + str(learningRate))
        classifier = MLPClassifier(learning_rate_init=learningRate, max_iter=maxIt)
        classifier.fit(X_train, y_train)
        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)

        mlpTrainingScoreList.append(trainingScore)
        mlpTestingScoreList.append(testingScore)
        print(trainingScore)

    icaTrainingScoreList = []
    icaTestingScoreList = []
    dIca = FastICA(whiten=True, n_components = 8)
    fittedIca = dIca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(fittedIca, y, test_size=0.33, random_state=0)
    for learningRate in learningRates:
        print("Running ICA MLP Learning rate: " + str(learningRate))
        classifier = MLPClassifier(learning_rate_init=learningRate, max_iter=maxIt)
        classifier.fit(X_train, y_train)
        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)

        icaTrainingScoreList.append(trainingScore)
        icaTestingScoreList.append(testingScore)
        print(trainingScore)

    pcaTrainingScoreList = []
    pcaTestingScoreList = []
    dPca = PCA()
    fittedPca = dPca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(fittedPca, y, test_size=0.33, random_state=0)
    for learningRate in learningRates:
        print("Running PCA MLP Learning rate: " + str(learningRate))
        classifier = MLPClassifier(learning_rate_init=learningRate, max_iter=maxIt)
        classifier.fit(X_train, y_train)
        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)

        pcaTrainingScoreList.append(trainingScore)
        pcaTestingScoreList.append(testingScore)
        print(trainingScore)

    ldaTrainingScoreList = []
    ldaTestingScoreList = []
    dLd = LinearDiscriminantAnalysis()
    dLdFitted = dLd.fit_transform(X, y)
    # testFitted = dLd.fit_transform(X_test, y_test)
    X_train, X_test, y_train, y_test = train_test_split(dLdFitted, y, test_size=0.33, random_state=0)
    for learningRate in learningRates:
        print("Running LDA MLP Learning rate: " + str(learningRate))
        classifier = MLPClassifier(learning_rate_init=learningRate, max_iter=maxIt)
        classifier.fit(X_train, y_train)
        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)
        # testingScore = classifier.score(testFitted, y_test)

        ldaTrainingScoreList.append(trainingScore)
        ldaTestingScoreList.append(testingScore)
        print(trainingScore)

    rpTrainingScoreList = []
    rpTestingScoreList = []
    dSp = SparseRandomProjection(n_components=8)
    dSpFitted = dSp.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(dSpFitted, y, test_size=0.33, random_state=0)
    for learningRate in learningRates:
        print("Running RP MLP Learning rate: " + str(learningRate))
        classifier = MLPClassifier(learning_rate_init=learningRate, max_iter=maxIt)
        classifier.fit(X_train, y_train)
        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)

        rpTrainingScoreList.append(trainingScore)
        rpTestingScoreList.append(testingScore)
        print(trainingScore)


    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    titleStr = "Dimensionaliity Reduction - Learning Rate vs Training Set Score"
    plt.title(titleStr)
    plt.xlabel("Learning Rate")
    plt.ylabel("Training Set Accuracy")
    plt.plot(learningRates, mlpTrainingScoreList, '-o', label="MLP Classifier Training Accuracy")
    plt.plot(learningRates, icaTrainingScoreList, '-o', label="ICA NN Training Accuracy")
    plt.plot(learningRates, pcaTrainingScoreList, '-o', label="PCA NN Training Accuracy")
    plt.plot(learningRates, ldaTrainingScoreList, '-o', label="LDA NN Training Accuracy")
    plt.plot(learningRates, rpTrainingScoreList, '-o', label="RP NN Training Accuracy")
    plt.legend(loc='best')
    saveFigPath = "figures/" + str(dataset) + "-DR-Learning-RatevsTraining-SetScore.png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    titleStr = "Dimensionaliity Reduction - Learning Rate vs Testing Set Score"
    plt.title(titleStr)
    plt.xlabel("Learning Rate")
    plt.ylabel("Testing Set Accuracy")
    plt.plot(learningRates, mlpTestingScoreList, '-o', label="MLP Classifier Testing Accuracy")
    plt.plot(learningRates, icaTestingScoreList, '-o', label="ICA NN Testing Accuracy")
    plt.plot(learningRates, pcaTestingScoreList, '-o', label="PCA NN Testing Accuracy")
    plt.plot(learningRates, ldaTestingScoreList, '-o', label="LDA NN Testing Accuracy")
    plt.plot(learningRates, rpTestingScoreList, '-o', label="RP NN Training Accuracy")
    plt.legend(loc='best')
    saveFigPath = "figures/" + str(dataset) + "-DR-Learning-RatevsTesting-SetScore.png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    kmTrainingScoreList = []
    kmTestingScoreList = []
    kMeansModel = KMeans(n_clusters=X.shape[1], max_iter=3000, n_init=50)
    yPredicted = kMeansModel.fit_predict(X)
    ypO = yPredicted.shape[0]
    kMeanX = np.ndarray((ypO, 1))
    kMeanX[:,0] = yPredicted
    # yp1 = y.shape[0]
    # kMeanY = np.ndarray((yp1, 1))
    # kMeanY[:,0] = y
    X_train, X_test, y_train, y_test = train_test_split(kMeanX, y, test_size=0.33, random_state=0)
    for learningRate in learningRates:
        print("Running KM MLP Learning rate: " + str(learningRate))
        classifier = MLPClassifier(learning_rate_init=learningRate, max_iter=maxIt)
        classifier.fit(X_train, y_train)
        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)

        kmTrainingScoreList.append(trainingScore)
        kmTestingScoreList.append(testingScore)
        print(trainingScore)

    mlpTrainingScoreList.clear()
    mlpTestingScoreList.clear()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    for learningRate in learningRates:
        print("Running MLP Learning rate: " + str(learningRate))
        classifier = MLPClassifier(learning_rate_init=learningRate, max_iter=maxIt)
        classifier.fit(X_train, y_train)
        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)

        mlpTrainingScoreList.append(trainingScore)
        mlpTestingScoreList.append(testingScore)
        print(trainingScore)

    emTrainingScoreList = []
    emTestingScoreList = []
    expMaxModel = GaussianMixture(n_components=X.shape[1], max_iter=3000, n_init=50)
    # expMaxModel.fit(X)
    # yPredicted = expMaxModel.predict(X, y)
    yPredicted = expMaxModel.fit_predict(X)
    ypO = yPredicted.shape[0]
    emX = np.ndarray((ypO, 1))
    emX[:,0] = yPredicted
    # yp1 = y.shape[0]
    # kMeanY = np.ndarray((yp1, 1))
    # kMeanY[:,0] = y
    X_train, X_test, y_train, y_test = train_test_split(emX, y, test_size=0.33, random_state=0)
    for learningRate in learningRates:
        print("Running EM MLP Learning rate: " + str(learningRate))
        classifier = MLPClassifier(learning_rate_init=learningRate, max_iter=maxIt)
        classifier.fit(X_train, y_train)
        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)

        emTrainingScoreList.append(trainingScore)
        emTestingScoreList.append(testingScore)
        print(trainingScore)

    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    titleStr = "Clustering Reduction - Learning Rate vs Scores"
    plt.title(titleStr)
    plt.xlabel("Learning Rate")
    plt.ylabel("Scores")
    plt.plot(learningRates, kmTrainingScoreList, '-o', label="KMeans Training Accuracy")
    plt.plot(learningRates, kmTestingScoreList, '-o', label="KMeans Testing Accuracy")
    plt.plot(learningRates, emTrainingScoreList, '-o', label="Expectation Maximization Training Accuracy")
    plt.plot(learningRates, emTestingScoreList, '-o', label="Expectation Maximization Testing Accuracy")
    plt.plot(learningRates, mlpTrainingScoreList, '-o', label="MLP Training Accuracy")
    plt.plot(learningRates, mlpTestingScoreList, '-o', label="MLP Testing Accuracy")
    plt.legend(loc='best')
    saveFigPath = "figures/" + str(dataset) + "-Clustering-Learning-RatevsScore.png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()


    mlpTrainingScoreList.clear()
    mlpTestingScoreList.clear()
    icaTrainingScoreList.clear()
    icaTestingScoreList.clear()
    pcaTrainingScoreList.clear()
    pcaTestingScoreList.clear()
    ldaTrainingScoreList.clear()
    ldaTestingScoreList.clear()
    rpTrainingScoreList.clear()
    rpTestingScoreList.clear()
    kmTrainingScoreList.clear()
    kmTestingScoreList.clear()
    emTrainingScoreList.clear()
    emTestingScoreList.clear()

    mlpRunTime = []
    maxIters = [500, 1000, 1500, 2000]
    learningRate = 0.0003

    for maxIt in maxIters:
        print("Running MLP MAX iter: " + str(maxIt))
        classifier = MLPClassifier(learning_rate_init=learningRate, max_iter=maxIt)
        classifier.fit(X_train, y_train)
        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)

        mlpTrainingScoreList.append(trainingScore)
        mlpTestingScoreList.append(testingScore)
        print(trainingScore)

    dIca = FastICA(whiten=True, n_components = 8)
    fittedIca = dIca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(fittedIca, y, test_size=0.33, random_state=0)
    for maxIt in maxIters:
        print("Running ICA MLP MAX iter: " + str(maxIt))
        classifier = MLPClassifier(learning_rate_init=learningRate, max_iter=maxIt)
        classifier.fit(X_train, y_train)
        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)

        icaTrainingScoreList.append(trainingScore)
        icaTestingScoreList.append(testingScore)
        print(trainingScore)

    dPca = PCA()
    fittedPca = dPca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(fittedPca, y, test_size=0.33, random_state=0)
    for maxIt in maxIters:
        print("Running PCA MLP MAX iter: " + str(maxIt))
        classifier = MLPClassifier(learning_rate_init=learningRate, max_iter=maxIt)
        classifier.fit(X_train, y_train)
        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)

        pcaTrainingScoreList.append(trainingScore)
        pcaTestingScoreList.append(testingScore)
        print(trainingScore)

    dLd = LinearDiscriminantAnalysis()
    dLdFitted = dLd.fit_transform(X, y)
    # testFitted = dLd.fit_transform( X_test, y_test)
    X_train, X_test, y_train, y_test = train_test_split(dLdFitted, y, test_size=0.33, random_state=0)
    for maxIt in maxIters:
        print("Running LDA MLP MAX iter: " + str(maxIt))
        classifier = MLPClassifier(learning_rate_init=learningRate, max_iter=maxIt)
        classifier.fit(X_train, y_train)
        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)

        ldaTrainingScoreList.append(trainingScore)
        ldaTestingScoreList.append(testingScore)
        print(trainingScore)

    dSp = SparseRandomProjection(n_components=8)
    dSpFitted = dSp.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(dSpFitted, y, test_size=0.33, random_state=0)
    for maxIt in maxIters:
        print("Running RP MLP MAX iter: " + str(maxIt))
        classifier = MLPClassifier(learning_rate_init=learningRate, max_iter=maxIt)
        classifier.fit(X_train, y_train)
        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)

        rpTrainingScoreList.append(trainingScore)
        rpTestingScoreList.append(testingScore)
        print(trainingScore)



    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    titleStr = "Dimensionaliity Reduction - Max Iteration vs Training Set Score"
    plt.title(titleStr)
    plt.xlabel("Max Iterations")
    plt.ylabel("Training Set Accuracy")
    plt.plot(maxIters, mlpTrainingScoreList, '-o', label="MLP Classifier Training Accuracy")
    plt.plot(maxIters, icaTrainingScoreList, '-o', label="ICA NN Training Accuracy")
    plt.plot(maxIters, pcaTrainingScoreList, '-o', label="PCA NN Training Accuracy")
    plt.plot(maxIters, ldaTrainingScoreList, '-o', label="LDA NN Training Accuracy")
    plt.plot(maxIters, rpTrainingScoreList, '-o', label="RP NN Training Accuracy")
    plt.legend(loc='best')
    saveFigPath = "figures/" + str(dataset) + "-RD-Max-IterationsvsTraining-SetScore.png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    titleStr = "Dimensionaliity Reduction - Max Iteration vs Testing Set Score"
    plt.title(titleStr)
    plt.xlabel("Max Iterations")
    plt.ylabel("Testing Set Accuracy")
    plt.plot(maxIters, mlpTestingScoreList, '-o', label="MLP Classifier Testing Accuracy")
    plt.plot(maxIters, icaTestingScoreList, '-o', label="ICA NN Testing Accuracy")
    plt.plot(maxIters, pcaTestingScoreList, '-o', label="PCA NN Testing Accuracy")
    plt.plot(maxIters, ldaTestingScoreList, '-o', label="LDA NN Testing Accuracy")
    plt.plot(maxIters, rpTestingScoreList, '-o', label="RP NN Testing Accuracy")
    plt.legend(loc='best')
    saveFigPath = "figures/" + str(dataset) + "-RD-Max-IterstionsvsTesting-SetScore.png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()
    # X.shape[1]
    kMeansModel = KMeans(n_clusters=20, max_iter=3000, n_init=50)
    yPredicted = kMeansModel.fit_predict(X)
    ypO = yPredicted.shape[0]
    kMeanX = np.ndarray((ypO, 1))
    kMeanX[:,0] = yPredicted
    # yp1 = y.shape[0]
    # kMeanY = np.ndarray((yp1, 1))
    # kMeanY[:,0] = y
    X_train, X_test, y_train, y_test = train_test_split(kMeanX, y, test_size=0.33, random_state=0)
    for maxIt in maxIters:
        print("Running KM MLP MAX iter: " + str(maxIt))
        classifier = MLPClassifier(learning_rate_init=learningRate, max_iter=maxIt)
        classifier.fit(X_train, y_train)
        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)

        kmTrainingScoreList.append(trainingScore)
        kmTestingScoreList.append(testingScore)
        print(trainingScore)

    mlpTrainingScoreList.clear()
    mlpTestingScoreList.clear()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    for maxIt in maxIters:
        print("Running MLP MAX iter: " + str(maxIt))
        classifier = MLPClassifier(learning_rate_init=learningRate, max_iter=maxIt)
        classifier.fit(X_train, y_train)
        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)

        mlpTrainingScoreList.append(trainingScore)
        mlpTestingScoreList.append(testingScore)
        print(trainingScore)

    expMaxModel = GaussianMixture(n_components=20, max_iter=3000, n_init=50)
    # expMaxModel.fit(X)
    # yPredicted = expMaxModel.predict(X, y)
    yPredicted = expMaxModel.fit_predict(X)
    ypO = yPredicted.shape[0]
    emX = np.ndarray((ypO, 1))
    emX[:,0] = yPredicted
    # yp1 = y.shape[0]
    # kMeanY = np.ndarray((yp1, 1))
    # kMeanY[:,0] = y
    X_train, X_test, y_train, y_test = train_test_split(emX, y, test_size=0.33, random_state=0)
    for maxIt in maxIters:
        print("Running EM MLP MAX iter: " + str(maxIt))
        classifier = MLPClassifier(learning_rate_init=learningRate, max_iter=maxIt)
        classifier.fit(X_train, y_train)
        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)

        emTrainingScoreList.append(trainingScore)
        emTestingScoreList.append(testingScore)
        print(trainingScore)

    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    titleStr = "Clustering Reduction - Max Iter vs Scores"
    plt.title(titleStr)
    plt.xlabel("Max Iterations")
    plt.ylabel("Scores")
    plt.plot(maxIters, kmTrainingScoreList, '-o', label="KMeans Training Accuracy")
    plt.plot(maxIters, kmTestingScoreList, '-o', label="KMeans Testing Accuracy")
    plt.plot(maxIters, emTrainingScoreList, '-o', label="Expectation Maximization Training Accuracy")
    plt.plot(maxIters, emTestingScoreList, '-o', label="Expectation Maximization Testing Accuracy")
    plt.plot(maxIters, mlpTrainingScoreList, '-o', label="MLP Training Accuracy")
    plt.plot(maxIters, mlpTestingScoreList, '-o', label="MLP Testing Accuracy")
    plt.legend(loc='best')
    saveFigPath = "figures/" + str(dataset) + "-Clustering-Max-Iter-vsScore.png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    # plt.style.use('seaborn-poster') #sets the size of the charts
    # plt.style.use('ggplot')
    # titleStr = "Max Iteration vs Run Time"
    # plt.title(titleStr)
    # plt.xlabel("Max Iterations")
    # plt.ylabel("Run Time")
    # plt.plot(maxIters, mlpRunTime, '-o', label="MLP Run Time")
    # plt.plot(maxIters, rhcRunTime, '-o', label="RHC Run Time")
    # plt.plot(maxIters, saRunTime, '-o', label="SA Run Time")
    # plt.plot(maxIters, gaRunTime, '-o', label="GA Run Time")
    # plt.plot(maxIters, gdRunTime, '-o', label="Gradient Descent Run Time")
    # plt.legend(loc='best')
    # saveFigPath = "figures/Max-IterstionsvsRun-Time.png"
    # plt.savefig(saveFigPath)
    # plt.clf()
    # plt.cla()
    # plt.close()


def rocCurve1(X, y, X_train, X_test, y_train, y_test, dataset):
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
    # nn_model1 = mlrose.NeuralNetwork(hidden_nodes=None, activation='relu', \
    # algorithm='random_hill_climb', max_iters=maxIt, bias=True, is_classifier=True, \
    # learning_rate=0.09, early_stopping=False, clip_max=10000000000.0, restarts=0,\
    # pop_size=200, mutation_prob=0.1, max_attempts=10, random_state=None)
    # nn_model1.fit(X_train, y_train)
    # calcPlot(nn_model1, "deeppink", "RHC")
    # nn_model1 = mlrose.NeuralNetwork(hidden_nodes=None, activation='relu', \
    # algorithm='simulated_annealing', max_iters=maxIt, bias=True, is_classifier=True, \
    # learning_rate=1, early_stopping=False, clip_max=10000000000.0, restarts=0,\
    # pop_size=200, mutation_prob=0.1, max_attempts=10, random_state=None)
    # nn_model1.fit(X_train, y_train)
    # calcPlot(nn_model1, "navy", "SA")
    # nn_model1 = mlrose.NeuralNetwork(hidden_nodes=None, activation='relu', \
    # algorithm='genetic_alg', max_iters=maxIt, bias=True, is_classifier=True, \
    # learning_rate=0.02, early_stopping=False, clip_max=10000000000.0, restarts=0,\
    # pop_size=200, mutation_prob=0.1, max_attempts=10, random_state=None)
    # nn_model1.fit(X_train, y_train)
    # calcPlot(nn_model1, "aqua", "GA")
    # nn_model1 = mlrose.NeuralNetwork(hidden_nodes=None, activation='relu', \
    # algorithm='gradient_descent', max_iters=maxIt, bias=True, is_classifier=True, \
    # learning_rate=0.03, early_stopping=False, clip_max=10000000000.0, restarts=0,\
    # pop_size=200, mutation_prob=0.1, max_attempts=10, random_state=None)
    # nn_model1.fit(X_train, y_train)
    # calcPlot(nn_model1, "darkorange", "Gradient Descent")
    print("ROC curve MLP")
    classifier = MLPClassifier(learning_rate_init=0.0004)
    classifier.fit(X_train, y_train)
    calcPlot(classifier, "cornflowerblue", "MLP")

    print("ROC curve ICA")
    dIca = FastICA(whiten=True, n_components = 8)
    fittedIca = dIca.fit_transform(X_train)
    classifier = MLPClassifier(learning_rate_init=0.0004)
    classifier.fit(fittedIca, y_train)
    calcPlot(classifier, "deeppink", "ICA")

    print("ROC curve PCA")
    dPca = PCA()
    fittedPca = dPca.fit_transform(X_train)
    classifier = MLPClassifier(learning_rate_init=0.0004)
    classifier.fit(fittedPca, y_train)
    calcPlot(classifier, "darkorange", "PCA")

    print("ROC curve LDA")
    dLd = LinearDiscriminantAnalysis()
    dLdFitted = dLd.fit_transform(X_train, y_train)
    testFitted = dLd.fit_transform(X_test, y_test)
    classifier = MLPClassifier(learning_rate_init=0.0004)
    classifier.fit(dLdFitted, y_train)
    calcPlot(classifier, "aqua", "LDA")

    print("ROC curve RP")
    dSp = SparseRandomProjection(n_components=8)
    dSpFitted = dSp.fit_transform(X_train)
    classifier = MLPClassifier(learning_rate_init=0.0004)
    classifier.fit(dSpFitted, y_train)
    calcPlot(classifier, "navy", "RP")


    plt.plot([0, 1], [0, 1], lw=lw, color='black', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Dimensionality Reduction - Receiver operating characteristic example')
    plt.legend(loc="lower right")
    saveFigPath = "figures/" + str(dataset) + "-DM-ROC-curve.png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    kMeansModel = KMeans(n_clusters=20, max_iter=3000, n_init=50)
    yPredicted = kMeansModel.fit_predict(X)
    ypO = yPredicted.shape[0]
    kMeanX = np.ndarray((ypO, 1))
    kMeanX[:,0] = yPredicted
    # yp1 = y.shape[0]
    # kMeanY = np.ndarray((yp1, 1))
    # kMeanY[:,0] = y
    X_train, X_test, y_train, y_test = train_test_split(kMeanX, y, test_size=0.33, random_state=0)
    classifier = MLPClassifier(learning_rate_init=0.0004)
    classifier.fit(X_train, y_train)
    calcPlot(classifier, "navy", "KMeans NN")

    print("ROC curve MLP")
    classifier = MLPClassifier(learning_rate_init=0.0004)
    classifier.fit(X_train, y_train)
    calcPlot(classifier, "cornflowerblue", "MLP")

    expMaxModel = GaussianMixture(n_components=20, max_iter=3000, n_init=50)
    # expMaxModel.fit(X)
    # yPredicted = expMaxModel.predict(X, y)
    yPredicted = expMaxModel.fit_predict(X)
    ypO = yPredicted.shape[0]
    emX = np.ndarray((ypO, 1))
    emX[:,0] = yPredicted
    # yp1 = y.shape[0]
    # kMeanY = np.ndarray((yp1, 1))
    # kMeanY[:,0] = y
    X_train, X_test, y_train, y_test = train_test_split(emX, y, test_size=0.33, random_state=0)
    classifier = MLPClassifier(learning_rate_init=0.0004)
    classifier.fit(X_train, y_train)
    calcPlot(classifier, "deeppink", "Expectation Maximization NN")

    plt.plot([0, 1], [0, 1], lw=lw, color='black', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Clustering Reduction NN - Receiver operating characteristic example')
    plt.legend(loc="lower right")
    saveFigPath = "figures/" + str(dataset) + "-CLNN-ROC-curve.png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

def main():
    print("### RUNNING NN ###")
    diabetesData = pd.read_csv("diabetes.csv")
    diabetesData = pd.DataFrame(diabetesData.values, columns=list(diabetesData))

    X = diabetesData.values[:,0:8]
    y = diabetesData.values[:,8]

    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    NNetwork1(X, y, X_train, X_test, y_train, y_test, "diabetes")
    rocCurve1(X, y, X_train, X_test, y_train, y_test, "diabetes")


    # heartData = pd.read_csv("heart.csv")
    # heartData = pd.DataFrame(heartData.values, columns=list(heartData))

    # X = heartData.values[:,0:13]
    # y = heartData.values[:,13]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    # NNetwork1(X, y, X_train, X_test, y_train, y_test, "heart")
    # rocCurve1(X, y, X_train, X_test, y_train, y_test, "heart")

    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target

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

    # NNetwork1(X, y, X_train, X_test, y_train, y_test, "2")
    # rocCurve1(X, y, X_train, X_test, y_train, y_test, "2")

    # # print(y_train)

    # # SVM2(X, y, X_train, X_test, y_train, y_test)
    # # DT2(X, y, X_train, X_test, y_train, y_test)
    # # KNN2(X, y, X_train, X_test, y_train, y_test)
    # # ADABoost2(X, y, X_train, X_test, y_train, y_test)
    # NNetwork1(X, y, X_train, X_test, y_train, y_test)
    # # rocCurve2(X, y, X_train, X_test, y_train, y_test)

if __name__== "__main__":
    main()
