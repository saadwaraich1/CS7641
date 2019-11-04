import matplotlib.pyplot as plt
import time
import matplotlib.style as style
import random
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, v_measure_score, silhouette_score
from sklearn.mixture import GaussianMixture

def plot(title, xLabel, yLabel, xInput, yInputs, legends, path):
    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    for idx, yInput in enumerate(yInputs):
        plt.plot(xInput, yInput, '-o', label=legends[idx])

    plt.legend(loc='best')
    plt.savefig(path)
    plt.clf()
    plt.cla()
    plt.close()

def plotBar(title, xLabel, yLabel, xInput, yInput, path):
    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.bar(x=xInput, height=yInput)
    # plt.legend(loc='best')
    plt.savefig(path)
    plt.clf()
    plt.cla()
    plt.close()

def processICA(X, y, dataset):

    dIca = FastICA(whiten=True, n_components = X.shape[1])
    fittedIca = dIca.fit_transform(X)
    results = []
    dims = fittedIca.shape[1]

    for i in range(dims):
        results.append(abs(kurtosis(fittedIca[:,i])))

    results.sort()
    # kurts = []
    # for i, kurt in enumerate(results):
    #     kurts.append((i, kurt))

    # kurts.sort(key=lambda x: x[1])

    dimensions = [x for x, _ in enumerate(results)]
    saveFigPath = "figures/" + str(dataset) + "-ICAKurtosis.png"
    plotBar("ICA Kurtosis vs Features", "Features", "Kurtosis", dimensions, results, saveFigPath)

    dIca = FastICA(whiten=True, n_components = None)
    fittedIca = dIca.fit_transform(X)

    kMeansSilhouette = []
    kMeansVmeasures = []
    kMeansHomogeneity = []
    kMeansSS = []
    kMeansRunTimes = []

    expMaxSilhouette = []
    expMaxVmeasures = []
    expMaxHomogeneity = []
    expMaxLH = []
    expMaxRunTimes = []

    clusterSizes = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for clusterSize in clusterSizes:
        print("Running KMeans - cluster size: " + str(clusterSize))
        startTime = time.time()
        kMeansModel = KMeans(n_clusters=clusterSize, max_iter=3000, n_init=50)
        yPredicted = kMeansModel.fit_predict(fittedIca)
        endTime = time.time()

        kMeansRunTimes.append(endTime-startTime)
        kMeansHomogeneity.append(homogeneity_score(y, yPredicted))
        kMeansVmeasures.append(v_measure_score(y, yPredicted))
        kMeansSilhouette.append(silhouette_score(X=fittedIca, labels=yPredicted, metric='euclidean', sample_size=X.shape[0]))
        kMeansSS.append(kMeansModel.score(fittedIca))

        print("Running Expectation Maximization - cluster size: " + str(clusterSize))
        startTime = time.time()
        expMaxModel = GaussianMixture(n_components=clusterSize, max_iter=3000, n_init=50)
        yPredicted = expMaxModel.fit_predict(fittedIca)
        endTime = time.time()

        expMaxRunTimes.append(endTime-startTime)
        expMaxHomogeneity.append(homogeneity_score(y, yPredicted))
        expMaxVmeasures.append(v_measure_score(y, yPredicted))
        expMaxSilhouette.append(silhouette_score(X=fittedIca, labels=yPredicted, metric='euclidean', sample_size=fittedIca.shape[0]))
        expMaxLH.append(expMaxModel.score(fittedIca))

    # saveFigPath = "figures/" + str(dataset) + "-ICA-KMEM-RunTimeVsClusterSize.png"
    # plot(title="Run Time vs Cluster Size", xLabel="Cluster Size", yLabel="Run Time", xInput=clusterSizes, \
    # yInputs=[kMeansRunTimes, expMaxRunTimes], legends=["KMeans Run Time", "Expectation Maximization Run Time"], \
    # path=saveFigPath)

    saveFigPath = "figures/" + str(dataset) + "-ICA-KMeans-SumSquareVsClusterSize.png"
    plot(title="ICA - KMeans Sum of Squares vs Cluster Size", xLabel="Cluster Size", yLabel="Sum of Squares", xInput=clusterSizes, \
    yInputs=[kMeansSS], legends=["Sum of square error for each cluster size"], path=saveFigPath)

    saveFigPath = "figures/" + str(dataset) + "-ICA-EM-LikelihoodVsClusterSize.png"
    plot(title="ICA - Expectation Maximization Log Likelihood vs Cluster Size", xLabel="Cluster Size", yLabel="Log Likelihood", \
    xInput=clusterSizes, yInputs=[expMaxLH], legends=["Log Likelihood for each cluster size"], path=saveFigPath)

    saveFigPath = "figures/" + str(dataset) + "-ICA-KMEM-ScoresVsClusterSize.png"
    plot(title="ICA - Scores vs Cluster Size", xLabel="Cluster Size", yLabel="Scores", xInput=clusterSizes, \
    yInputs=[kMeansHomogeneity, kMeansVmeasures, kMeansSilhouette, expMaxHomogeneity, expMaxVmeasures, expMaxSilhouette], \
    legends=["KMeans Homogeneity score", "KMeans V measure score", "KMeans Silhouette score", "Expectation Maximization Homogeneity score", \
    "Expectation Maximization V measure score", "Expectation Maximization Silhouette score"], path=saveFigPath)

    print("KMeans ICA Clustering")
    dIca = FastICA(whiten=True, n_components = X.shape[1])
    fittedIca = dIca.fit_transform(X)
    kMeansModel = KMeans(n_clusters=2, max_iter=100, n_init=1)
    yPredicted = kMeansModel.fit_predict(fittedIca)
    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    plt.title("KMeans ICA clusters")
    if dataset == "heart":
        plt.xlabel("Resting Blood Pressure")
        plt.ylabel("Serum Cholestoral in mg/dl")
        plt.scatter(X[:,3], X[:,4], c=yPredicted, alpha = 0.9)
    else:
        plt.xlabel("Glucose")
        plt.ylabel("BMI")
        plt.scatter(X[:,1], X[:,5], c=yPredicted, alpha = 0.9)
    saveFigPath = "figures/" + str(dataset) + "-KMeans-ICA-clusters.png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    print("KMeans EM Clustering")
    dIca = FastICA(whiten=True, n_components = X.shape[1])
    fittedIca = dIca.fit_transform(X)
    expMaxModel = GaussianMixture(n_components=2, max_iter=100, n_init=1)
    yPredicted = expMaxModel.fit_predict(fittedIca)
    plt.style.use('seaborn-poster') #sets the size of the charts
    plt.style.use('ggplot')
    plt.title("Expectation Maximization ICA clusters")
    if dataset == "heart":
        plt.xlabel("Resting Blood Pressure")
        plt.ylabel("Serum Cholestoral in mg/dl")
        plt.scatter(X[:,3], X[:,4], c=yPredicted, alpha = 0.9)
    else:
        plt.xlabel("Glucose")
        plt.ylabel("BMI")
        plt.scatter(X[:,1], X[:,5], c=yPredicted, alpha = 0.9)
    saveFigPath = "figures/" + str(dataset) + "-EM-ICA-clusters.png"
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

def main():
    print("### RUNNING ICA ###")
    diabetesData = pd.read_csv("diabetes.csv")
    diabetesData = pd.DataFrame(diabetesData.values, columns=list(diabetesData))

    X = diabetesData.values[:,0:8]
    y = diabetesData.values[:,8]

    processICA(X, y, "diabetes")
    # plt.style.use('seaborn-poster') #sets the size of the charts
    # plt.style.use('ggplot')
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    # NNetwork1(X, y, X_train, X_test, y_train, y_test)
    # rocCurve1(X, y, X_train, X_test, y_train, y_test)

    # X = pd.read_csv("mnist_train.csv").values
    # y = pd.read_csv("mnist_test.csv").values

    # X_train = X[0:60000,1:]
    # y_train = X[0:60000,0]

    # X_train = X[0:5000,1:]
    # y_train = X[0:5000,0]

    # y_train = y_train == 7

    # # X_test = y[:10000, 1:]
    # # y_test = y[:10000, 0]
    # X_test = y[:1000, 1:]
    # y_test = y[:1000, 0]

    # y_test = y_test == 7

    # X = X[0:6000, 1:]
    # y = X[0:6000, 0] == 7

    heartData = pd.read_csv("heart.csv")
    heartData = pd.DataFrame(heartData.values, columns=list(heartData))

    X = heartData.values[:,0:13]
    y = heartData.values[:,13]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    processICA(X, y, "heart")
    # processICA(X, y)


    # # print(y_train)

    # # SVM2(X, y, X_train, X_test, y_train, y_test)
    # # DT2(X, y, X_train, X_test, y_train, y_test)
    # # KNN2(X, y, X_train, X_test, y_train, y_test)
    # # ADABoost2(X, y, X_train, X_test, y_train, y_test)
    # NNetwork1(X, y, X_train, X_test, y_train, y_test)
    # # rocCurve2(X, y, X_train, X_test, y_train, y_test)

if __name__== "__main__":
    main()
