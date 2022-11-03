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

data = pd.read_csv("lab5ABT.txt", sep='\t', lineterminator='\r')

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
plt.plot(time, rpm)
# plt.show()




# ########## Question 3 ##########
# def averageProperties(rpm1, rp1):
#     for i in range(len(rpm)):
#         if rpm[i] > 46000 and rpm[i] << 50000:
#             averageRPM = 