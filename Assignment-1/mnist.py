import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.style as style
import scikitplot as skplt
import random
import pandas as pd
import numpy as np

def SVM2(X, y, X_train, X_test, y_train, y_test):
    # SVM code for different degrees of polynomial graph
    trainingScoreList = []
    testingScoreList = []
    crossValScoreList = []
    for degree in range(1,6):
        # classifier = svm.SVC(max_iter=1, kernel='poly', degree=degree, gamma='auto', random_state=0)
        classifier = svm.SVC(C=1.0, kernel='poly', degree=degree, gamma='auto',
          coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,
          class_weight=None, verbose=False, max_iter=1000, decision_function_shape=None,
          random_state=None)

        classifier.fit(X_train, y_train)

        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)
        crossValTestScore = cross_val_score(classifier, X_train, y_train, cv=5)
        crossValTestScore = sum(crossValTestScore)/5

        trainingScoreList.append(trainingScore)
        testingScoreList.append(testingScore)
        crossValScoreList.append(crossValTestScore)

    degrees = [1,2,3,4,5]
    plotModelVariabilityCurve("SVM", "degree", degrees, trainingScoreList, testingScoreList, crossValScoreList, "2")

    # SVM code for different sizes of training data set
    dataLen = len(X)
    trainingScoreList.clear()
    testingScoreList.clear()
    crossValScoreList.clear()
    for percent in [0.4,0.6,0.8,1]:
        #@percentX_train, percentX_test, percentY_train, percentY_test = train_test_split(diabetesData.data, diabetesData.target, test_size=0.33, random_state=0)
        percentX_train = X_train[0:int(dataLen*percent), :]
        percentY_train = y_train[0:int(dataLen*percent)]
        # percentX_train, percentX_test, percentY_train, percentY_test = train_test_split(percentX, percentY, test_size=0.33, random_state=0)
        # classifier = svm.SVC(max_iter=1, kernel='poly', degree=1, gamma='auto', random_state=0)
        classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',
          coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,
          class_weight=None, verbose=False, max_iter=1000, decision_function_shape=None,
          random_state=None)

        classifier.fit(percentX_train, percentY_train)

        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)
        crossValTestScore = cross_val_score(classifier, X_train, y_train, cv=5)
        crossValTestScore = sum(crossValTestScore)/5

        trainingScoreList.append(trainingScore)
        testingScoreList.append(testingScore)
        crossValScoreList.append(crossValTestScore)

    percents = ["40%", "60%", "80%", "100%"]
    plotModelLearningCurve("SVM", "Training Size", percents, trainingScoreList, testingScoreList, crossValScoreList, "2")

def DT2(X, y, X_train, X_test, y_train, y_test):
    # Decision tree code for different degrees of polynomial graph
    trainingScoreList = []
    testingScoreList = []
    crossValScoreList = []
    for treeDepth in [1,5,9,13,17]:
        classifier = DecisionTreeClassifier(random_state=0, max_depth=treeDepth)

        classifier.fit(X_train, y_train)

        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)
        crossValTestScore = cross_val_score(classifier, X_train, y_train, cv=5)
        crossValTestScore = sum(crossValTestScore)/5

        trainingScoreList.append(trainingScore)
        testingScoreList.append(testingScore)
        crossValScoreList.append(crossValTestScore)

    treeDepth = [1,5,9,13,17]
    plotModelVariabilityCurve("Decision Tree", "Depth", treeDepth, trainingScoreList, testingScoreList, crossValScoreList, "2")

    # Decision tree code for different sizes of training data set
    dataLen = len(X)
    trainingScoreList.clear()
    testingScoreList.clear()
    crossValScoreList.clear()
    for percent in [0.4,0.6,0.8,1]:
        percentX_train = X_train[0:int(dataLen*percent), :]
        percentY_train = y_train[0:int(dataLen*percent)]
        classifier = DecisionTreeClassifier(random_state=0, max_depth=10)

        classifier.fit(percentX_train, percentY_train)

        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)
        crossValTestScore = cross_val_score(classifier, X_train, y_train, cv=5)
        crossValTestScore = sum(crossValTestScore)/5

        trainingScoreList.append(trainingScore)
        testingScoreList.append(testingScore)
        crossValScoreList.append(crossValTestScore)

    percents = ["40%", "60%", "80%", "100%"]
    plotModelLearningCurve("Decision Tree", "Training Size", percents, trainingScoreList, testingScoreList, crossValScoreList, "2")

def KNN2(X, y, X_train, X_test, y_train, y_test):
    # Decision tree code for different degrees of polynomial graph
    trainingScoreList = []
    testingScoreList = []
    crossValScoreList = []
    for n_neighbors in [1,5,9,13,17]:
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

        classifier.fit(X_train, y_train)

        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)
        crossValTestScore = cross_val_score(classifier, X_train, y_train, cv=5)
        crossValTestScore = sum(crossValTestScore)/5

        trainingScoreList.append(trainingScore)
        testingScoreList.append(testingScore)
        crossValScoreList.append(crossValTestScore)

    n_neighbors = [1,5,9,13,17]
    plotModelVariabilityCurve("KNN", "No. of Neighbors", n_neighbors, trainingScoreList, testingScoreList, crossValScoreList, "2")

    # Decision tree code for different sizes of training data set
    dataLen = len(X)
    trainingScoreList.clear()
    testingScoreList.clear()
    crossValScoreList.clear()
    for percent in [0.4,0.6,0.8,1]:
        percentX_train = X_train[0:int(dataLen*percent), :]
        percentY_train = y_train[0:int(dataLen*percent)]
        classifier = KNeighborsClassifier(n_neighbors=2)

        classifier.fit(percentX_train, percentY_train)

        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)
        crossValTestScore = cross_val_score(classifier, X_train, y_train, cv=5)
        crossValTestScore = sum(crossValTestScore)/5

        trainingScoreList.append(trainingScore)
        testingScoreList.append(testingScore)
        crossValScoreList.append(crossValTestScore)

    percents = ["40%", "60%", "80%", "100%"]
    plotModelLearningCurve("KNN", "Training Size", percents, trainingScoreList, testingScoreList, crossValScoreList, "2")

def ADABoost2(X, y, X_train, X_test, y_train, y_test):
    # Decision tree code for different degrees of polynomial graph
    trainingScoreList = []
    testingScoreList = []
    crossValScoreList = []
    for n_estimators in [20,30,40,50,60]:
        classifier = DecisionTreeClassifier(random_state=0, max_depth=4)
        classifier = AdaBoostClassifier(base_estimator=classifier, n_estimators=n_estimators)
        classifier.fit(X_train, y_train)

        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)
        crossValTestScore = cross_val_score(classifier, X_train, y_train, cv=5)
        crossValTestScore = sum(crossValTestScore)/5

        trainingScoreList.append(trainingScore)
        testingScoreList.append(testingScore)
        crossValScoreList.append(crossValTestScore)

    n_estimators = [20,30,40,50,60]
    plotModelVariabilityCurve("ADABoost", "Estimators", n_estimators, trainingScoreList, testingScoreList, crossValScoreList, "2")

    # Decision tree code for different sizes of training data set
    dataLen = len(X)
    trainingScoreList.clear()
    testingScoreList.clear()
    crossValScoreList.clear()
    for percent in [0.4,0.6,0.8,1]:
        percentX_train = X_train[0:int(dataLen*percent), :]
        percentY_train = y_train[0:int(dataLen*percent)]
        classifier = DecisionTreeClassifier(random_state=0, max_depth=4)
        classifier = AdaBoostClassifier(base_estimator=classifier, n_estimators=40)

        classifier.fit(percentX_train, percentY_train)

        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)
        crossValTestScore = cross_val_score(classifier, X_train, y_train, cv=5)
        crossValTestScore = sum(crossValTestScore)/5

        trainingScoreList.append(trainingScore)
        testingScoreList.append(testingScore)
        crossValScoreList.append(crossValTestScore)

    percents = ["40%", "60%", "80%", "100%"]
    plotModelLearningCurve("ADABoost", "Training Size", percents, trainingScoreList, testingScoreList, crossValScoreList, "2")

def NNetwork2(X, y, X_train, X_test, y_train, y_test):
    # Decision tree code for different degrees of polynomial graph
    trainingScoreList = []
    testingScoreList = []
    crossValScoreList = []
    for learningRate in [0.001,0.002,0.003,0.004,0.005]:
        classifier = MLPClassifier(learning_rate_init=learningRate)
        classifier.fit(X_train, y_train)

        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)
        crossValTestScore = cross_val_score(classifier, X_train, y_train, cv=5)
        crossValTestScore = sum(crossValTestScore)/5

        trainingScoreList.append(trainingScore)
        testingScoreList.append(testingScore)
        crossValScoreList.append(crossValTestScore)

    learningRate = [0.001,0.002,0.003,0.004,0.005]
    plotModelVariabilityCurve("Multi-layer Perceptron classifier", "Learning Rate", learningRate, trainingScoreList, testingScoreList, crossValScoreList, "2")

    # Decision tree code for different sizes of training data set
    dataLen = len(X)
    trainingScoreList.clear()
    testingScoreList.clear()
    crossValScoreList.clear()
    for percent in [0.4,0.6,0.8,1]:
        percentX_train = X_train[0:int(dataLen*percent), :]
        percentY_train = y_train[0:int(dataLen*percent)]
        classifier = MLPClassifier(learning_rate_init=0.002)

        classifier.fit(percentX_train, percentY_train)

        trainingScore = classifier.score(X_train, y_train)
        testingScore = classifier.score(X_test, y_test)
        crossValTestScore = cross_val_score(classifier, X_train, y_train, cv=5)
        crossValTestScore = sum(crossValTestScore)/5

        trainingScoreList.append(trainingScore)
        testingScoreList.append(testingScore)
        crossValScoreList.append(crossValTestScore)

    percents = ["40%", "60%", "80%", "100%"]
    plotModelLearningCurve("Multi-layer Perceptron classifier", "Training Size", percents, trainingScoreList, testingScoreList, crossValScoreList, "2")

def plotModelVariabilityCurve(classifierName, variableName, variableList, trainingScoreList, testingScoreList, crossValScoreList, dataset):
    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    titleStr = "Variability Curve - " + str(classifierName)
    plt.title(titleStr)
    plt.xlabel(variableName)
    plt.ylabel("Score")
    plt.plot(variableList, trainingScoreList, '-o', color = "red", label="Training Set Score")
    plt.plot(variableList, testingScoreList, '-o', color = "blue", label="Testing Set Score")
    plt.plot(variableList, crossValScoreList, '-o', color = "green", label="CV Testing Set Score")
    plt.legend(loc='best')
    saveFigPath = "figures/" + dataset + titleStr + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

def plotModelLearningCurve(classifierName, variableName, variableList, trainingScoreList, testingScoreList, crossValScoreList, dataset):
    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    titleStr = "Learning Curve - " + str(classifierName)
    plt.title(titleStr)
    plt.xlabel(variableName)
    plt.ylabel("Score")
    plt.plot(variableList, trainingScoreList, '-o', color = "red", label="Training Set Score")
    plt.plot(variableList, testingScoreList, '-o', color = "blue", label="Testing Set Score")
    plt.plot(variableList, crossValScoreList, '-o', color = "green", label="CV Testing Set Score")
    plt.legend(loc='best')
    saveFigPath = "figures/" + dataset + titleStr + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

# def rocCurve2(X, y, X_train, X_test, y_train, y_test, diabetesData):
#     lw = 2
#     def calcPlot(classifier, color, clfName):
#         # Compute ROC curve and ROC area for each class
#         fpr = dict()
#         tpr = dict()
#         roc_auc = dict()


#         # Compute micro-average ROC curve and ROC area
#         fpr["micro"], tpr["micro"], _ = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1])
#         roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#         plt.plot(fpr["micro"], tpr["micro"], color=color,
#                 lw=lw, label=clfName + (' (area = %0.2f)' % roc_auc["micro"]))
#     # SVM
#     plt.figure()
#     classifier = svm.SVC(kernel='linear', degree=3, gamma='auto', probability=True)
#     classifier.fit(X_train, y_train)
#     calcPlot(classifier, "deeppink", "SVM")
#     classifier = DecisionTreeClassifier(random_state=0, max_depth=10)
#     classifier.fit(X_train, y_train)
#     calcPlot(classifier, "navy", "Decision Tree")
#     classifier = KNeighborsClassifier(n_neighbors=2)
#     classifier.fit(X_train, y_train)
#     calcPlot(classifier, "aqua", "KNN")
#     classifier = DecisionTreeClassifier(random_state=0, max_depth=4)
#     classifier = AdaBoostClassifier(base_estimator=classifier, n_estimators=40)
#     classifier.fit(X_train, y_train)
#     calcPlot(classifier, "darkorange", "ADA-Boost")
#     classifier = MLPClassifier(learning_rate_init=0.0004)
#     classifier.fit(X_train, y_train)
#     calcPlot(classifier, "cornflowerblue", "MLP")
#     plt.plot([0, 1], [0, 1], lw=lw, color='black', linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic example')
#     plt.legend(loc="lower right")
#     saveFigPath = "figures/2ROC-curve.png"
#     plt.savefig(saveFigPath)
#     plt.clf()
#     plt.cla()
#     plt.close()

def rocCurve2(X, y, X_train, X_test, y_train, y_test):
    lw = 2
    n_classes = 10
    def calcPlot(classifier, color, clfName):
        plt.figure()
        # # Compute ROC curve and ROC area for each class
        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()

        # for i in range(n_classes):
        #     # Compute micro-average ROC curve and ROC area
        #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], classifier.predict_proba(X_test[:, i])[:, 1])
        #     roc_auc[i] = auc(fpr[i], tpr[i])
        # # plt.plot(fpr[i], tpr[i], color=color,
        # #         lw=lw, label=clfName + (' (area = %0.2f)' % roc_auc["micro"]))
        # colors = ["deeppink", "navy", "aqua", "darkorange", "cornflowerblue", "magenta", "red", "green",  "yellow", "blue"]
        # for i, color in zip(range(n_classes), colors):
        #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
        #             label=clfName + ('ROC curve of class {0} (area = {1:0.2f})'
        #             ''.format(i, roc_auc[i])))

        # plt.plot([0, 1], [0, 1], lw=lw, color='black', linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        # plt.legend(loc="lower right")
        y_true = y_test
        y_probas = classifier.predict_proba(X_test)
        # y_probas = classifier.predict(X_test)[:, 1]
        skplt.metrics.plot_roc_curve(y_true, y_probas)
        # plt.show()
        plt.title(clfName + " ROC curve ")
        saveFigPath = "figures/" + clfName + "ROC-curve.png"
        plt.savefig(saveFigPath)
        plt.clf()
        plt.cla()
        plt.close()


    # #SVM
    # y_train = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    classifier = DecisionTreeClassifier(random_state=0, max_depth=10)
    classifier.fit(X_train, y_train)
    calcPlot(classifier, "navy", "Decision Tree")
    classifier = DecisionTreeClassifier(random_state=0, max_depth=4)
    classifier = AdaBoostClassifier(base_estimator=classifier, n_estimators=40)
    classifier.fit(X_train, y_train)
    calcPlot(classifier, "darkorange", "ADA-Boost")
    classifier = MLPClassifier(learning_rate_init=0.0004)
    classifier.fit(X_train, y_train)
    calcPlot(classifier, "cornflowerblue", "MLP")
    classifier = svm.SVC(kernel='linear', degree=3, gamma='auto', probability=True)
    classifier.fit(X_train, y_train)
    calcPlot(classifier, "deeppink", "SVM")
    classifier = KNeighborsClassifier(n_neighbors=2)
    classifier.fit(X_train, y_train)
    calcPlot(classifier, "aqua", "KNN")


def main():

    X = pd.read_csv("mnist_train.csv").values
    y = pd.read_csv("mnist_test.csv").values

    X_train = X[0:60000,1:]
    y_train = X[0:60000,0]

    X_test = y[:10000, 1:]
    y_test = y[:10000, 0]

    SVM2(X, y, X_train, X_test, y_train, y_test)
    DT2(X, y, X_train, X_test, y_train, y_test)
    KNN2(X, y, X_train, X_test, y_train, y_test)
    ADABoost2(X, y, X_train, X_test, y_train, y_test)
    NNetwork2(X, y, X_train, X_test, y_train, y_test)
    rocCurve2(X, y, X_train, X_test, y_train, y_test)


if __name__== "__main__":
    main()
