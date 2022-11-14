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
plt.xlabel("Time [s]")
plt.ylabel("RPM [rpm]")
plt.title("Time [s] vs Jet Engine RPM [rpm]")
# plt.savefig("Question_2.png", dpi = 300)


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
print(df) 


def kelvinConvert(T):
    return T + 273.15

def interpolation(h1, h2, T, T1, T2):
    return h1 + (T - T1)*(h2 - h1)/(T2 - T1)

print(kelvinConvert(t1rpmAverageMax), kelvinConvert(t2rpmAverageMax), kelvinConvert(t3rpmAverageMax), kelvinConvert(t4rpmAverageMax))

###### Question 3 Part 2 ######
h1Start =  interpolation(295.17, 300.19, kelvinConvert(t1AverageStart), 295, 300) 
h2Start = interpolation(295.17, 300.19, kelvinConvert(t2AverageStart), 295, 300)  
h3Start = interpolation(295.17, 300.19, kelvinConvert(t3AverageStart), 295, 300) 
h4Start = interpolation(295.17, 300.19, kelvinConvert(t4AverageStart), 295, 300) 

h1FortyEight =  interpolation(295.17, 300.19, kelvinConvert(t1AverageFourtyEight), 295, 300) 
h2FortyEight = interpolation(325.31, 330.34, kelvinConvert(t2AverageFourtyEight), 325, 330)  
h3FortyEight = interpolation(800.03, 810.99, kelvinConvert(t3AverageFourtyEight), 780, 790) 
h4FortyEight = interpolation(472.24, 482.49, kelvinConvert(t4AverageFourtyEight), 470, 480)

h1FiftyEight =  interpolation(295.17, 300.19, kelvinConvert(t1AverageFiftyEight), 295, 300) 
h2FiftyEight = interpolation(380.77, 390.88, kelvinConvert(t2AverageFiftyEight), 380, 390)  
h3FiftyEight = interpolation(932.93, 955.38, kelvinConvert(t3AverageFiftyEight), 900, 920) 
h4FiftyEight = interpolation(586.04, 596.52, kelvinConvert(t4AverageFiftyEight), 580, 590)


h1SixtyEight =  interpolation(295.17, 300.19, kelvinConvert(t1AverageSixtyEight), 295, 300) 
h2SixtyEight = interpolation(411.12, 421.26, kelvinConvert(t2AverageSixtyEight), 410, 420)  
h3SixtyEight = interpolation(1023.25, 1046.04, kelvinConvert(t3AverageSixtyEight), 980, 1000) 
h4SixtyEight = interpolation(607.02, 617.53, kelvinConvert(t4AverageSixtyEight), 610, 620)

h1Max =  interpolation(295.17, 300.19, kelvinConvert(t1rpmAverageMax), 295, 300) 
h2Max = interpolation(451.80, 462.02, kelvinConvert(t2rpmAverageMax), 450, 460)  
h3Max = interpolation(1114.86, 1137.89, kelvinConvert(t3rpmAverageMax), 1060, 1080) 
h4Max = interpolation(628.07, 638.63, kelvinConvert(t4rpmAverageMax), 620, 630)

Pamb = 97.6 # kPa
def staticPressure(P):
    return Pamb - P

P1StartStatic = staticPressure(p1AverageStart)
P2StartStatic = staticPressure(p2AverageStart)
P3StartStatic = staticPressure(p3AverageStart)
P4StartStatic = staticPressure(p4AverageStart)

P1FourtyEightStatic = staticPressure(p1AverageFourtyEight)
P2FourtyEightStatic = staticPressure(p2AverageFourtyEight)
P3FourtyEightStatic = staticPressure(p3AverageFourtyEight)
P4FourtyEightStatic = staticPressure(p4AverageFourtyEight)

P1FiftyEightStatic = staticPressure(p1AverageFiftyEight)
P2FiftyEightStatic = staticPressure(p2AverageFiftyEight)
P3FiftyEightStatic = staticPressure(p3AverageFiftyEight)
P4FiftyEightStatic = staticPressure(p4AverageFiftyEight)

P1SixtyEightStatic = staticPressure(p1AverageSixtyEight)
P2SixtyEightStatic = staticPressure(p2AverageSixtyEight)
P3SixtyEightStatic = staticPressure(p3AverageSixtyEight)
P4SixtyEightStatic = staticPressure(p4AverageSixtyEight)

P1MaxStatic = staticPressure(p1rpmAverageMax)
P2MaxStatic = staticPressure(p2rpmAverageMax)
P3MaxStatic = staticPressure(p3rpmAverageMax)
P4MaxStatic = staticPressure(p4rpmAverageMax)

df3 = pd.DataFrame(np.array([[h1Start, h2Start, h3Start, h4Start, P1StartStatic, P2StartStatic, P3StartStatic, P4StartStatic],
    [h1FortyEight, h2FortyEight, h3FortyEight, h4FortyEight, P1FourtyEightStatic, P2FourtyEightStatic, P3FourtyEightStatic, P4FourtyEightStatic],
    [h1FiftyEight, h2FiftyEight, h3FiftyEight, h4FiftyEight, P1FiftyEightStatic, P2FiftyEightStatic, P3FiftyEightStatic, P4FiftyEightStatic],
    [h1SixtyEight, h2SixtyEight, h3SixtyEight, h4SixtyEight, P1SixtyEightStatic, P2SixtyEightStatic, P3SixtyEightStatic, P4SixtyEightStatic],
    [h1Max, h2Max, h3Max, h4Max, P1MaxStatic, P2MaxStatic, P3MaxStatic, P4MaxStatic]]))
df3.columns = ["h1", "h2", "h3", "h4", "P1", "P2", "P3", "P4"]
df3 = df3.round(decimals = 4)
# print(df3.to_latex(index=False))


