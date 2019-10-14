# Author: Saad Waraich
#GTID: 903459227

import time
import mlrose
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style as style
import random
import pandas as pd

lengths=[10, 20, 40, 60, 80, 100]
avgAcross=10

print("SOLVING ContinuousPeaks")
print("Get fitness for 100 iters on all algos")

itersList = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
problem=mlrose.DiscreteOpt(length=50, fitness_fn=mlrose.ContinuousPeaks())

fitnessRHCAll = []
fitnessSAAll = []
fitnessGAAll = []
fitnessMIMICAll = []

fitnessRHCMean = []
fitnessSAMean = []
fitnessGAMean = []
fitnessMIMICMean = []

fitnessRHCFilter = []
fitnessSAFilter = []
fitnessGAFilter = []
fitnessMIMICFilter = []

fitnessRHCMean.append(0)
fitnessSAMean.append(0)
fitnessGAMean.append(0)
fitnessMIMICMean.append(0)

for run in range(avgAcross):
    print("RUN - " + str(run+1) + " for 100 iterations graph")
    bestStateRHC, bestFitnessRHC, fitnessCurveRHC=mlrose.random_hill_climb(problem=problem, max_iters=101, max_attempts=20, restarts=2, curve=True)
    bestStateSA, bestFitnessSA, fitnessCurveSA=mlrose.simulated_annealing(problem=problem, max_iters=101, max_attempts=20, curve=True)
    bestStateGA, bestFitnessGA, fitnessCurveGA=mlrose.genetic_alg(problem=problem, max_iters=101,  pop_size=40, curve=True)
    bestStateMIMIC, bestFitnessMIMIC, fitnessCurveMIMIC=mlrose.mimic(problem=problem, max_iters=101, pop_size=40, curve=True)

    fitnessRHCAll.append(fitnessCurveRHC)
    fitnessSAAll.append(fitnessCurveSA)
    fitnessGAAll.append(fitnessCurveGA)
    fitnessMIMICAll.append(fitnessCurveMIMIC)

for i in range(min(map(len, fitnessRHCAll))):
    for j in [0,1,2,3,4,5,6,7,8,9]:
        fitnessRHCFilter.append(fitnessRHCAll[j][i])

    fitnessRHCMean.append(np.mean(fitnessRHCFilter))
    fitnessRHCFilter.clear()

for i in range(min(map(len, fitnessSAAll))):
    for j in [0,1,2,3,4,5,6,7,8,9]:
        fitnessSAFilter.append(fitnessSAAll[j][i])

    fitnessSAMean.append(np.mean(fitnessSAFilter))
    fitnessSAFilter.clear()

for i in range(min(map(len, fitnessGAAll))):
    for j in [0,1,2,3,4,5,6,7,8,9]:
        fitnessGAFilter.append(fitnessGAAll[j][i])

    fitnessGAMean.append(np.mean(fitnessGAFilter))
    fitnessGAFilter.clear()

for i in range(min(map(len, fitnessMIMICAll))):
    for j in [0,1,2,3,4,5,6,7,8,9]:
        fitnessMIMICFilter.append(fitnessMIMICAll[j][i])

    fitnessMIMICMean.append(np.mean(fitnessMIMICFilter))
    fitnessMIMICFilter.clear()

plt.style.use('seaborn-poster')
plt.style.use('ggplot')
titleStr="Continuous Peaks - Mean Fitness Value vs No. of Iterations"
plt.title(titleStr)
plt.xlabel("No. of Iterations")
plt.ylabel("Mean Fitness Value")
# plt.plot(variableList, trainingScoreList, '-o', color="red", label="Training Set Score")
# plt.plot(variableList, testingScoreList, '-o', color="blue", label="Testing Set Score")
# plt.plot(variableList, crossValScoreList, '-o', color="green", label="CV Testing Set Score")
# pd.DataFrame({
#     "RHC": fitnessRHCMean,
#     "SA": fitnessSAMean,
#     "GA": fitnessGAMean,
#     "MIMIC": fitnessMIMICMean,
    
# }, index=sizes).plot(marker='o', ax=ax)
plt.plot(range(0,len(fitnessRHCMean)), fitnessRHCMean, '-o', label="randomized hill climbing")
plt.plot(range(0,len(fitnessSAMean)), fitnessSAMean, '-o', label="simulated annealing")
plt.plot(range(0,len(fitnessGAMean)), fitnessGAMean, '-o', label="a genetic algorithm")
plt.plot(range(0,len(fitnessMIMICMean)), fitnessMIMICMean, '-o', label="MIMIC")
plt.legend(loc='best')
saveFigPath="figures/" + "ContinuousPeaks-fitnesVsIter-" + ".png"
plt.savefig(saveFigPath)
plt.clf()
plt.cla()
plt.close()



print("Starting RHC")
fitnessValMeanRHC=[]
timeToRunMeanRHC=[]

for length in lengths:

    problem=mlrose.DiscreteOpt(length=length, fitness_fn=mlrose.ContinuousPeaks())
    
    fitnessVal=[]
    timeToRun=[]

    for run in range(avgAcross):
        print("Running RHC for length: " + str(length) + ", run: " + str(run))
        startTime=time.time()
        bestState, bestFitness, fitnessCurve=mlrose.random_hill_climb(problem=problem, max_attempts=20, restarts=2, curve=True)
        timeTaken=time.time() - startTime

        fitnessVal.append(bestFitness)
        timeToRun.append(timeTaken)

    fitnessValMeanRHC.append(np.mean(fitnessVal))
    timeToRunMeanRHC.append(np.mean(timeToRun))


print("Starting SA")
fitnessValMeanSA=[]
timeToRunMeanSA=[]

for length in lengths:

    problem=mlrose.DiscreteOpt(length=length, fitness_fn=mlrose.ContinuousPeaks())

    fitnessVal=[]
    timeToRun=[]

    for run in range(avgAcross):
        print("Running SA for length: " + str(length) + ", run: " + str(run))
        startTime=time.time()
        bestState, bestFitness, fitnessCurve=mlrose.simulated_annealing(problem=problem, max_attempts=20, curve=True)
        timeTaken=time.time() - startTime

        fitnessVal.append(bestFitness)
        timeToRun.append(timeTaken)

    fitnessValMeanSA.append(np.mean(fitnessVal))
    timeToRunMeanSA.append(np.mean(timeToRun))


print("Starting GA")
fitnessValMeanGA=[]
timeToRunMeanGA=[]

for length in lengths:

    problem=mlrose.DiscreteOpt(length=length, fitness_fn=mlrose.ContinuousPeaks())

    fitnessVal=[]
    timeToRun=[]

    for run in range(avgAcross):
        print("Running SA for length: " + str(length) + ", run: " + str(run))
        startTime=time.time()
        bestState, bestFitness, fitnessCurve=mlrose.genetic_alg(problem=problem,  pop_size=40, curve=True)
        timeTaken=time.time() - startTime

        fitnessVal.append(bestFitness)
        timeToRun.append(timeTaken)

    fitnessValMeanGA.append(np.mean(fitnessVal))
    timeToRunMeanGA.append(np.mean(timeToRun))


print("Starting MIMIC")
fitnessValMeanMIMIC=[]
timeToRunMeanMIMIC=[]

for length in lengths:

    problem=mlrose.DiscreteOpt(length=length, fitness_fn=mlrose.ContinuousPeaks())

    fitnessVal=[]
    timeToRun=[]

    for run in range(avgAcross):
        print("Running MIMIC for length: " + str(length) + ", run: " + str(run))
        startTime=time.time()
        bestState, bestFitness, fitnessCurve=mlrose.mimic(problem=problem, pop_size=40, curve=True)
        timeTaken=time.time() - startTime

        fitnessVal.append(bestFitness)
        timeToRun.append(timeTaken)

    fitnessValMeanMIMIC.append(np.mean(fitnessVal))
    timeToRunMeanMIMIC.append(np.mean(timeToRun))


plt.style.use('seaborn-poster')
plt.style.use('ggplot')
titleStr="Continuous Peaks - Mean Fitness Value vs Problem Length"
plt.title(titleStr)
plt.xlabel("Problem Length")
plt.ylabel("Mean Fitness Value")
# plt.plot(variableList, trainingScoreList, '-o', color="red", label="Training Set Score")
# plt.plot(variableList, testingScoreList, '-o', color="blue", label="Testing Set Score")
# plt.plot(variableList, crossValScoreList, '-o', color="green", label="CV Testing Set Score")
plt.plot(lengths, fitnessValMeanRHC, '-o', label="randomized hill climbing")
plt.plot(lengths, fitnessValMeanSA, '-o', label="simulated annealing")
plt.plot(lengths, fitnessValMeanGA, '-o', label="a genetic algorithm")
plt.plot(lengths, fitnessValMeanMIMIC, '-o', label="MIMIC")
plt.legend(loc='best')
saveFigPath="figures/" + "ContinuousPeaks-fitnessVal-" + ".png"
plt.savefig(saveFigPath)
plt.clf()
plt.cla()
plt.close()

plt.style.use('seaborn-poster')
plt.style.use('ggplot')
titleStr="Continuous Peaks - Mean Run Time vs Problem Length"
plt.title(titleStr)
plt.xlabel("Problem Length")
plt.ylabel("Mean Run Time")
# plt.plot(variableList, trainingScoreList, '-o', color="red", label="Training Set Score")
# plt.plot(variableList, testingScoreList, '-o', color="blue", label="Testing Set Score")
# plt.plot(variableList, crossValScoreList, '-o', color="green", label="CV Testing Set Score")
plt.plot(lengths, timeToRunMeanRHC, '-o', label="randomized hill climbing")
plt.plot(lengths, timeToRunMeanSA, '-o', label="simulated annealing")
plt.plot(lengths, timeToRunMeanGA, '-o', label="a genetic algorithm")
plt.plot(lengths, timeToRunMeanMIMIC, '-o', label="MIMIC")
plt.legend(loc='best')
saveFigPath="figures/" + "ContinuousPeaks-timeToRun-" + ".png"
plt.savefig(saveFigPath)
plt.clf()
plt.cla()
plt.close()

