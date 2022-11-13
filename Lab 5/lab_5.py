from cProfile import label
from operator import le
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import figure
from matplotlib import axes as ax
from mpl_toolkits import mplot3d
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import warnings
from scipy.integrate import solve_ivp
import math
import csv

import matplotlib.colors as mcolors

data = pd.read_csv("2022-10-30 Sample Data.txt", sep='\t', lineterminator='\r')

time = np.array(data["Time (sec)"].tolist())
t1 = np.array(data["T1 (C)"].tolist())
t2 = np.array(data["T2 (C)"].tolist())
t3 = np.array(data["T3 (C)"].tolist())
t4 = np.array(data["T4 (C)"].tolist())
p1 = np.array(data["P1 (kPa)"].tolist())
p2 = np.array(data["P1 (kPa)"].tolist())
p3 = np.array(data["P3 (kPa)"].tolist())
p4 = np.array(data["P4 (kPa)"].tolist())
fuelFLow = np.array(data["Fuel Flow  (L/hr)"].tolist())
rpm = np.array(data["RPM"].tolist())
thrust = np.array(data["Thrust (N)"].tolist())




########## Question 2 ##########
fourtyeightIndexes = []
for i in range(0, 1000):
    if rpm[i] > 47500 and rpm[i] < 48500:
        fourtyeightIndexes.append(i)

fiftyeightIndexes = []
for i in range(0, 1000):
    if rpm[i] > 57500 and rpm[i] < 58500:
        fiftyeightIndexes.append(i)

sixtyeightIndexes = []
for i in range(0, 1000):
    if rpm[i] > 67500 and rpm[i] < 68500:
        sixtyeightIndexes.append(i)

max = []
for i in range(0, 1300):
    if rpm[i] > 76300:
        max.append(i)


fig, ax = plt.subplots()
ax.plot(time, rpm, color='black')
ax.axvspan(0, 120, facecolor='indianred', alpha=.5)
ax.axvspan(fourtyeightIndexes[10], fourtyeightIndexes[-3], facecolor='indianred', alpha=.5)
ax.axvspan(fiftyeightIndexes[20], fiftyeightIndexes[-3], facecolor='indianred', alpha=.5)
ax.axvspan(sixtyeightIndexes[15], sixtyeightIndexes[-3], facecolor='indianred', alpha=.5)
ax.axvspan(max[0], max[-1], facecolor='indianred', alpha=.5)
# plt.show()


steadyStateFortyEight = fourtyeightIndexes[10:-3]
steadyStateFiftyEight = fiftyeightIndexes[20:-3]
steadyStateSixtyEight = sixtyeightIndexes[15:-3]
steadyMax = max[0:-1]





# ########## Question 3 ##########
#Tabulated  columns  should  be  RPM,  T1(C),  T2(C),  T3(C),  T4(C),  P1(kPa),  P2(kPa), P3(kPa), P4(kPa), Fuel Flow (L/hr),  and Thrust (N).  

# Average values at startup
def average(minIndex, maxIndex):
    rpmAverage = np.average(rpm[minIndex:maxIndex])
    t1Average = np.average(t1[minIndex:maxIndex])
    t2Average = np.average(t2[minIndex:maxIndex])
    t3Average = np.average(t3[minIndex:maxIndex])
    t4Average = np.average(t4[minIndex:maxIndex])
    p1Average = np.average(p1[minIndex:maxIndex])
    p2Average = np.average(p2[minIndex:maxIndex])
    p3Average = np.average(p3[minIndex:maxIndex])
    p4Average = np.average(p4[minIndex:maxIndex])
    fuelFlowAverage = np.average(fuelFLow[minIndex:maxIndex])
    thrustAverage = np.average(thrust[minIndex:maxIndex])

    return rpmAverage, t1Average, t2Average, t3Average, t4Average, p1Average, p2Average, p3Average, p4Average, fuelFlowAverage, thrustAverage

rpmAverageStart, t1AverageStart, t2AverageStart, t3AverageStart, t4AverageStart, p1AverageStart, p2AverageStart, p3AverageStart, p4AverageStart, fuelFlowAverageStart, thrustAverageStart = average(0, 120)

rpmAverageFourtyEight, t1AverageFourtyEight, t2AverageFourtyEight, t3AverageFourtyEight, t4AverageFourtyEight, p1AverageFourtyEight, p2AverageFourtyEight, p3AverageFourtyEight, p4AverageFourtyEight, fuelFlowAverageFourtyEight, thrustAverageFourtyEight = average(steadyStateFortyEight[0], steadyStateFortyEight[-1])

rpmAverageFiftyEight, t1AverageFiftyEight, t2AverageFiftyEight, t3AverageFiftyEight, t4AverageFiftyEight, p1AverageFiftyEight, p2AverageFiftyEight, p3AverageFiftyEight, p4AverageFiftyEight, fuelFlowAverageFiftyEight, thrustAverageFiftyEight = average(steadyStateFiftyEight[0], steadyStateFiftyEight[-1])

rpmAverageSixtyEight, t1AverageSixtyEight, t2AverageSixtyEight, t3AverageSixtyEight, t4AverageSixtyEight, p1AverageSixtyEight, p2AverageSixtyEight, p3AverageSixtyEight, p4AverageSixtyEight, fuelFlowAverageSixtyEight, thrustAverageSixtyEight = average(steadyStateSixtyEight[0], steadyStateSixtyEight[-1])

rpmAverageMax, t1rpmAverageMax, t2rpmAverageMax, t3rpmAverageMax, t4rpmAverageMax, p1rpmAverageMax, p2rpmAverageMax, p3rpmAverageMax, p4rpmAverageMax, fuelFlowrpmAverageMax, thrustrpmAverageMax = average(steadyMax[0], steadyMax[-1])

df = pd.DataFrame(np.array([[rpmAverageStart, t1AverageStart, t2AverageStart, t3AverageStart, t4AverageStart, p1AverageStart, p2AverageStart, p3AverageStart, p4AverageStart, fuelFlowAverageStart, thrustAverageStart], [rpmAverageFourtyEight, t1AverageFourtyEight, t2AverageFourtyEight, t3AverageFourtyEight, t4AverageFourtyEight, p1AverageFourtyEight, p2AverageFourtyEight, p3AverageFourtyEight, p4AverageFourtyEight, fuelFlowAverageFourtyEight, thrustAverageFourtyEight], [rpmAverageFiftyEight, t1AverageFiftyEight, t2AverageFiftyEight, t3AverageFiftyEight, t4AverageFiftyEight, p1AverageFiftyEight, p2AverageFiftyEight, p3AverageFiftyEight, p4AverageFiftyEight, fuelFlowAverageFiftyEight, thrustAverageFiftyEight], [rpmAverageSixtyEight, t1AverageSixtyEight, t2AverageSixtyEight, t3AverageSixtyEight, t4AverageSixtyEight, p1AverageSixtyEight, p2AverageSixtyEight, p3AverageSixtyEight, p4AverageSixtyEight, fuelFlowAverageSixtyEight, thrustAverageSixtyEight], [rpmAverageMax, t1rpmAverageMax, t2rpmAverageMax, t3rpmAverageMax, t4rpmAverageMax, p1rpmAverageMax, p2rpmAverageMax, p3rpmAverageMax, p4rpmAverageMax, fuelFlowrpmAverageMax, thrustrpmAverageMax]]))
df.columns = ["Average RPM [rpm]", "Average T1 [C]", "Average T2 [C]", "Average T3 [C]", "Average T4 [C]", "Average P1 [kPa]", "Average P2 [kPa]", "Average P3 [kPa]", "Average P4 [kPa]","Average Fuel Flow [L/hr]", "Average Thrust [N]"]
df = df.round(decimals = 4)
print(df.to_latex(index=False))  