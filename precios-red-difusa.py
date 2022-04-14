# -*- coding: utf-8 -*-
"""
@author: Equipo 2
"""

import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.axes3d import get_test_data
from matplotlib import cm
import math 

#hola probando
#Data position in array
"""M1(0), DE1(1)
M2(2), DE2(3)
M3(4), DE3(5)
M4(6), DE4(7)
M5(8), DE5(9)
M6(10), DE6(11)

P1(12), Q1(13), R1(14)
P2(15), Q2(16), R2(17)
P3(18), Q3(19), R3(20)
P4(21), Q4(22), R4(23)
P5(24), Q5(25), R5(26)
P6(27), Q6(28), R6(29)
P7(30), Q7(31), R7(32)
P8(33), Q8(34), R8(35)
P9(36), Q9(37), R9(38)"""

M1, DE1 = 0, 8
M2, DE2 = 6, 8
M3, DE3 = 12, 4
M4, DE4 = 8.5, 2.5
M5, DE5 = 14, 4
M6, DE6 = 5.8, 0.78

I0, I1 = 1, 3
I2, I3 = 2, 2
I4, I5 = 1, 1
I6, I7 = 2, 1
I8, I9 = 4, 5
I10, I11 = 5, 5

P1, Q1, R1 = -19, 2, 2
P2, Q2, R2 = 1, 1, 1
P3, Q3, R3 = 1, -2, 1
P4, Q4, R4 = 2, 1, 1
P5, Q5, R5 = 1, 3, 1
P6, Q6, R6 = 2, 3, 5
P7, Q7, R7 = 2, 1, 5
P8, Q8, R8 = 6, 1, 5
P9, Q9, R9 = 6, 5, 5

I12, I13, I14 = 1, 1, 1
I15, I16, I17 = 1, 1, 1
I18, I19, I20 = 1, 1, 1
I21, I22, I23 = 1, 1, 1
I24, I25, I26 = 1, 1, 1
I27, I28, I29 = 1, 1, 1
I30, I31, I32 = 1, 1, 1
I33, I34, I35 = 1, 1, 1
I36, I37, I38 = 1, 1, 1

incProgress = 0
MF1, MF2, MF3, MF4, MF5, MF6, progress = [], [], [], [], [], [], []
POPSIZE, MUTATION_RATE, DOUBLE_MUTATION, CHILD_SIZE = 100, 10, 2, 200/20

Xfuzzyp = np.arange(1, 19, 0.1)
X = np.arange(1, 16, 1)
Y = np.arange(1, 13, 1)
Z = np.array([[.512,  .512,  .488,  .488,  .512,  .472,  .472,  .472,  .472,  .472,  .649,  .830], 
              [1.180, 1.180, 1.180, 1.100, 1.180, 1.180, 1.100, 1.100, 1.230, 1.370, 1.940, 1.940],  
              [1.230, 1.230, 1.230, 1.230, 1.230, 1.390, 1.390, 1.390, 1.150, 1.310, 1.630, 1.590],
              [1.180, 1.130, 1.060, 1.060, 1.030, 1.080, 1.080, 1.060, 1.030, 1.010,  .984, 1.080],
              [1.720, 1.830, 1.790, 1.610, 1.780, 1.780, 1.580, 1.580, 1.550, 1.290, 1.030, 1.030],
              [1.380, 1.380, 1.380, 1.380, 1.380, 1.380, 1.380, 1.300, 1.300, 1.210, 1.180, 1.180],
              [1.420, 1.420, 1.420, 1.460, 1.510, 1.460, 1.510, 1.560, 1.560, 1.560, 1.560, 1.560],
              [1.520, 1.590, 1.760, 1.760, 1.680, 1.760, 1.770, 1.770, 1.770, 1.570, 1.590, 1.570],
              [1.560, 1.560, 1.560, 1.560, 1.560, 1.660, 1.660, 1.660, 1.750, 1.440, 1.020, 1.020],
              [1.640, 1.660, 1.660, 1.620, 1.620, 1.600, 1.520, 1.520, 1.520, 1.520, 1.560, 1.560],
              [1.870, 1.870, 1.850, 1.940, 1.940, 2.010, 2.030, 2.030, 2.030, 2.040, 2.040, 2.040],
              [1.820, 1.820, 1.820, 1.820, 1.820, 1.820, 1.820, 1.820, 1.820, 1.820, 1.820, 1.820],
              [1.570, 2.030, 2.210, 1.840, 2.560, 2.900, 2.900, 2.900, 2.900, 2.900, 2.900, 2.900],
              [2.160, 2.160, 2.450, 2.480, 2.750, 2.750, 2.750, 2.750, 2.750, 2.750, 2.750, 2.750],
              [2.150, 2.150, 1.490, 1.440, 1.440, 1.440, 1.440, 1.440, 1.440, 1.390, 1.330, 1.320]])
              
Z1 = np.array([[ 512,  512,  541,  512,  536,  488,  488,  512,  512,  488,  488,  512,  472,  472,  472,  472,  472,  649,  830], 
              [ 850,  860, 1180, 1180, 1270, 1180, 1180, 1180, 1180, 1180, 1100, 1180, 1180, 1100, 1100, 1230, 1370, 1940, 1940], 
              [1070, 1070, 1070, 1070, 1150, 1150, 1150, 1230, 1230, 1230, 1230, 1230, 1390, 1390, 1390, 1150, 1310, 1630, 1590],
              [1180, 1180, 1180, 1180, 1180, 1180, 1180, 1180, 1130, 1060, 1060, 1030, 1080, 1080, 1060, 1030, 1010,  984, 1080],
              [1230, 1230, 1250, 1250, 1280, 1720, 1830, 1720, 1830, 1790, 1610, 1780, 1780, 1580, 1580, 1550, 1290, 1030, 1030],
              [1380, 1380, 1380, 1380, 1380, 1380, 1330, 1380, 1380, 1380, 1380, 1380, 1380, 1380, 1300, 1300, 1210, 1180, 1180],
              [1420, 1420, 1420, 1420, 1420, 1420, 1420, 1420, 1420, 1420, 1460, 1510, 1460, 1510, 1560, 1560, 1560, 1560, 1560],
              [1460, 1460, 1470, 1480, 1480, 1470, 1470, 1520, 1590, 1760, 1760, 1680, 1760, 1770, 1770, 1770, 1570, 1590, 1570],
              [1460, 1460, 1460, 1460, 1460, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1660, 1660, 1660, 1750, 1440, 1020, 1020],
              [1570, 1570, 1570, 1570, 1570, 1570, 1600, 1640, 1660, 1660, 1620, 1620, 1600, 1520, 1520, 1520, 1520, 1560, 1560],
              [1630, 1650, 1650, 1670, 1670, 1700, 1780, 1870, 1870, 1850, 1940, 1940, 2010, 2030, 2030, 2030, 2040, 2040, 2040],
              [1770, 1770, 1770, 1330, 1330, 1800, 1820, 1820, 1820, 1820, 1820, 1820, 1820, 1820, 1820, 1820, 1820, 1820, 1820],
              [1810, 1810, 1730, 1730, 1620, 1620, 1570, 1570, 2030, 2210, 1840, 2560, 2900, 2900, 2900, 2900, 2900, 2900, 2900],
              [2070, 2070, 2160, 2160, 2160, 2160, 2160, 2160, 2160, 2450, 2480, 2750, 2750, 2750, 2750, 2750, 2750, 2750, 2750],
              [2150, 2150, 2150, 2150, 2150, 2150, 2150, 2150, 2150, 1490, 1440, 1440, 1440, 1440, 1440, 1440, 1390, 1330, 1320],
              [2430, 2430, 2430, 1620, 1620, 1620, 1620, 1620, 1620, 1620, 1700, 1700, 1700, 1700, 1620, 1700, 1770, 1770, 1770],
              [2610,2610,2610,2640,2670,2720,2640,2550,2580,2560,2550,2460,2520,2510,2500,2480,2450,2450,2460],
              [2650,2650,2660,2660,2660,2660,2660,2710,2800,2850,2780,2850,3100,3180,3140,3340,3340,3340,3340],
              [2710,2740,2740,2740,2740,2800,2800,2800,2800,2800,2800,2800,2800,2770,2770,280,2800,2800,2990],
              [2750,2780,2800,2800,2800,2800,2800,2800,2800,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750],
              [2940,2950,2970,3050,3140,3200,3200,3210,3250,3200,3150,3100,3050,3070,3050,2940,2900,2900,2870],
              [3300,3340,3340,3390,3440,3440,3540,3590,3630,3640,3690,3720,3730,3740,3790,3830,3890,3890,3900],
              [3300,3300,3300,3020,2750,2460,2460,2980,2980,2980,2980,2160,2160,2160,2160,2160,2160,2160,2160],
              [3440,3440,3440,3440,3440,3540,3530,3530,3440,3490,3440,3440,3440,3440,3440,3340,3250,3150,3320],
              [3440,3440,3440,3440,3440,3540,3530,3530,3440,3490,3440,3440,3440,3440,3440,3340,3250,3150,3320],
              [3640,3640,3640,3640,3640,3640,3640,3640,3740,3740,3740,3740,3740,3740,3740,3570,3380,3250,3000],
              [3660,3660,3730,3730,3790,3790,3770,3840,3740,3740,3840,3730,3840,3840,3840,3860,3930,4110,4180],
              [3890,3890,3910,3910,3930,3930,3930,3930,3980,4110,4080,4080,4130,4110,4180,4180,4130,4160,4160],
              [3890,3890,3760,3760,3300,3640,3760,3760,3640,3640,3890,3890,3760,3810,3690,3340,3810,3810,3340],
              [4030,4030,4030,4110,4180,4230,4380,4430,4500,4480,4470,4520,4530,4550,4600,4530,4590,4620,4670],
              [4120,4130,4130,4120,4250,4250,4380,4430,5020,5020,5020,4700,4700,4700,4700,4700,5020,5020,5020],
              [4630,4630,4630,4630,4630,4630,4630,4630,4630,8610,8610,8610,8610,4650,4650,8610,7570,7570,7570],
              [5120,5120,5160,5210,5260,5290,5300,5360,5370,5400,5400,5410,5410,5450,5500,5560,5610,5660,5660],
              [5160,5160,5160,5160,5160,5160,5160,5160,5160,5160,5160,5160,5210,5310,5360,5360,5310,5400,5460],
              [5690,5690,5690,5690,5690,5690,5690,5830,5800,5700,5690,5690,5800,5800,5700,5750,5610,5610,5610],
              [7410,7380,7230,7380,6890,6790,6780,6570,6470,6590,6390,5950,6000,5950,5900,5710,5560,5700,5650],
              [7770,7770,7770,7770,7770,7770,7770,7770,7770,7770,7770,7770,7770,8780,9810,9180,9180,9180,9180],
              [8240,8240,8240,8720,8720,7160,7620,8660,8690,8720,8720,8410,8660,8660,8690,8720,8690,8660,8310],
              [8360,8360,8360,8360,8360,7860,7620,7770,7480,7770,7770,7620,7480,7620,7770,7480,7480,7480,7480],
              [8760,8760,8760,8760,8760,8760,8760,8760,8760,8760,8760,8760,8760,8760,8760,8070,7850,7820,7670],
              [9820,9820,9820,9820,9820,9820,9820,9830,12200,12200,14500,13700,14100,14100,14100,14100,13700,14100,14100],
              [10700,10700,10700,10600,10300,10300,10200,9960,9640,9530,9390,9350,9220,9050,9100,8360,8020,7870,8210],
              [21600,21400,21400,21400,21500,21400,21400,20600,20600,20600,20600,20600,20700,20600,20700,20600,20100,19700,19600]])

plt.ion()
fig = plt.figure(figsize=plt.figaspect(0.25))
fig.subplots_adjust(left=0.2, wspace=0.5)
ax = fig.add_subplot(1, 3, 1, projection='3d')
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)
#ax4 = fig.add_subplot(1, 4, 4)
Xp,Yp = np.meshgrid(np.array(Y),np.array(X))

def setInitPopulation():
    initPop = []
    for x in range(POPSIZE):
        subject = []
        for y in range(0, 39):
            tempVal = int(random.random() * 255) + 1
            subject.append(tempVal)
        initPop.append(subject)
    return initPop


def getCurveError(localPop):
    for a in range(0, POPSIZE):
        error = 0
        mf1, mf2, mf3, mf4, mf5, mf6 = [], [], [], [], [], []
        
        for k in range(0, 39):
            if(0 == localPop[a][k]):
                localPop[a][k] = 1
        
        for i in range(0,15):
            for j in range(0,12):

                mf1.append(math.exp((-(X[i]-(localPop[a][0]/I0))**2)/(2*(localPop[a][1]/I1)**2)))
                mf2.append(math.exp((-(X[i]-(localPop[a][2]/I2))**2)/(2*(localPop[a][3]/I3)**2)))
                mf3.append(math.exp((-(X[i]-(localPop[a][4]/I4))**2)/(2*(localPop[a][5]/I5)**2)))
                mf4.append(math.exp((-(Y[j]-(localPop[a][6]/I6))**2)/(2*(localPop[a][7]/I7)**2)))
                mf5.append(math.exp((-(Y[j]-(localPop[a][8]/I8))**2)/(2*(localPop[a][9]/I9)**2)))
                mf6.append(math.exp((-(Y[j]-(localPop[a][10]/I10))**2)/(2*(localPop[a][11]/I11)**2)))
                
                inf1=mf1[i]*mf4[j]
                inf2=mf1[i]*mf5[j]
                inf3=mf1[i]*mf6[j]
                inf4=mf2[i]*mf4[j]
                inf5=mf2[i]*mf5[j]
                inf6=mf2[i]*mf6[j]
                inf7=mf3[i]*mf4[j]
                inf8=mf3[i]*mf5[j]
                inf9=mf3[i]*mf6[j]
                
                p1 = (localPop[a][12]/I12)-19
                q3 = (localPop[a][19]/I19)-2
        
                reg1=inf1*(p1*X[i]+(localPop[a][13]/I13)*Y[j]+(localPop[a][14]/I14))
                reg2=inf2*((localPop[a][15]/I15)*X[i]+(localPop[a][16]/I16)*Y[j]+(localPop[a][17]/I17))
                reg3=inf1*((localPop[a][18]/I18)*X[i]+q3*Y[j]+(localPop[a][20]/I20))
                reg4=inf1*((localPop[a][21]/I21)*X[i]+(localPop[a][22]/I22)*Y[j]+(localPop[a][23]/I23))
                reg5=inf1*((localPop[a][24]/I24)*X[i]+(localPop[a][25]/I25)*Y[j]+(localPop[a][26]/I26))
                reg6=inf1*((localPop[a][27]/I27)*X[i]+(localPop[a][28]/I28)*Y[j]+(localPop[a][29]/I29))
                reg7=inf1*((localPop[a][30]/I30)*X[i]+(localPop[a][31]/I31)*Y[j]+(localPop[a][32]/I32))
                reg8=inf1*((localPop[a][33]/I33)*X[i]+(localPop[a][34]/I34)*Y[j]+(localPop[a][35]/I35))
                reg9=inf1*((localPop[a][36]/I36)*X[i]+(localPop[a][37]/I37)*Y[j]+(localPop[a][38]/I38))
        
                b=inf1+inf2+inf3+inf4+inf5+inf6+inf7+inf8+inf9
                total=reg1+reg2+reg3+reg4+reg5+reg6+reg7+reg8+reg9
                
                """TODO: Analyze if this action will throw an unwanted behaviour"""
                error += abs(Z[i][j] - (total/b))
                
        localPop[a].append(error)
        

def genSwitchMask(x, pos, arr1, arr2):
    pAltaP1, pBajaP1, pAltaP2, pBajaP2, mask = np.uint8(0), np.uint8(0), np.uint8(0), np.uint8(0), np.uint8(0)
    tempArr1 = [ y for y in arr1[:]]
    tempArr2 = [ y for y in arr2[:]]
    
    for i in range(0, x+1):
        mask += 1 << i
    pBajaP1 = mask & tempArr1[pos]
    pAltaP1 = (~mask) & tempArr1[pos]
    pBajaP2 = mask & tempArr2[pos]
    pAltaP2 = (~mask) & tempArr2[pos] 
    tempArr1[pos] = pAltaP1 | pBajaP2
    tempArr2[pos] = pAltaP2 | pBajaP1        
    
    return tempArr1, tempArr2
        

def genSwitch(arr1, arr2):
    x = int(random.random() * 311) 
    tempArr1 = [ z for z in arr1[:-1]]
    tempArr2 = [ z for z in arr2[:-1]]
    
    if ((x >= 0) and (x <= 7)):
        if(x != 7):
            tempArr1, tempArr2 = genSwitchMask(x, 38, tempArr1, tempArr2)
        else:
            tempArr1[38], tempArr2[38] = tempArr2[38], tempArr1[38]
    elif ((x >= 8) and (x <= 15)):
        if(x != 15):
            tempArr1, tempArr2 = genSwitchMask(x-8, 37, tempArr1, tempArr2)
            tempArr1[38], tempArr2[38] = tempArr2[38], tempArr1[38]
        else:
            tempArr1[37:39], tempArr2[37:39] = tempArr2[37:39], tempArr1[37:39]        
    elif ((x >= 16) and (x <= 23)):
        if(x != 23):
            tempArr1, tempArr2 = genSwitchMask(x-16, 36, tempArr1, tempArr2)
            tempArr1[37:39], tempArr2[37:39] = tempArr2[37:39], tempArr1[37:39]
        else:
            tempArr1[36:39], tempArr2[36:39] = tempArr2[36:39], tempArr1[36:39]
    elif ((x >= 24) and (x <= 31)):
        if(x != 31):
            tempArr1, tempArr2 = genSwitchMask(x-24, 35, tempArr1, tempArr2)
            tempArr1[36:39], tempArr2[36:39] = tempArr2[36:39], tempArr1[36:39]
        else:
            tempArr1[35:39], tempArr2[35:39] = tempArr2[35:39], tempArr1[35:39]   
    elif ((x >= 32) and (x <= 39)):
        if(x != 39):
            tempArr1, tempArr2 = genSwitchMask(x-32, 34, tempArr1, tempArr2)
            tempArr1[35:39], tempArr2[35:39] = tempArr2[35:39], tempArr1[35:39]
        else:
            tempArr1[34:39], tempArr2[34:39] = tempArr2[34:39], tempArr1[34:39] 
    elif ((x >= 40) and (x <= 47)):
        if(x != 47):
            tempArr1, tempArr2 = genSwitchMask(x-40, 33, tempArr1, tempArr2)
            tempArr1[34:39], tempArr2[34:39] = tempArr2[34:39], tempArr1[34:39]
        else:
            tempArr1[33:39], tempArr2[33:39] = tempArr2[33:39], tempArr1[33:39]   
    elif ((x >= 48) and (x <= 55)):
        if(x != 55):
            tempArr1, tempArr2 = genSwitchMask(x-48, 32, tempArr1, tempArr2)
            tempArr1[33:39], tempArr2[33:39] = tempArr2[33:39], tempArr1[33:39]
        else:
            tempArr1[32:39], tempArr2[32:39] = tempArr2[32:39], tempArr1[32:39] 
    elif ((x >= 56) and (x <= 63)):
        if(x != 63):
            tempArr1, tempArr2 = genSwitchMask(x-56, 31, tempArr1, tempArr2)
            tempArr1[32:39], tempArr2[32:39] = tempArr2[32:39], tempArr1[32:39]
        else:
            tempArr1[31:39], tempArr2[31:39] = tempArr2[31:39], tempArr1[31:39]         
    elif ((x >= 64) and (x <= 71)):
        if(x != 71):
            tempArr1, tempArr2 = genSwitchMask(x-64, 30, tempArr1, tempArr2)
            tempArr1[31:39], tempArr2[31:39] = tempArr2[31:39], tempArr1[31:39]
        else:
            tempArr1[30:39], tempArr2[30:39] = tempArr2[30:39], tempArr1[30:39]            
    elif ((x >= 72) and (x <= 79)):
        if(x != 79):
            tempArr1, tempArr2 = genSwitchMask(x-72, 29, tempArr1, tempArr2)
            tempArr1[30:39], tempArr2[30:39] = tempArr2[30:39], tempArr1[30:39]
        else:
            tempArr1[29:39], tempArr2[29:39] = tempArr2[29:39], tempArr1[29:39]
    elif ((x >= 80) and (x <= 87)):
        if(x != 87):
            tempArr1, tempArr2 = genSwitchMask(x-80, 28, tempArr1, tempArr2)
            tempArr1[29:39], tempArr2[29:39] = tempArr2[29:39], tempArr1[29:39]
        else:
            tempArr1[28:39], tempArr2[28:39] = tempArr2[28:39], tempArr1[28:39]         
    elif ((x >= 88) and (x <= 95)):
        if(x != 95):
            tempArr1, tempArr2 = genSwitchMask(x-88, 27, tempArr1, tempArr2)
            tempArr1[28:39], tempArr2[28:39] = tempArr2[28:39], tempArr1[28:39]
        else:
            tempArr1[27:39], tempArr2[27:39] = tempArr2[27:39], tempArr1[27:39]             
    elif ((x >= 96) and (x <= 103)):
        if(x != 103):
            tempArr1, tempArr2 = genSwitchMask(x-96, 26, tempArr1, tempArr2)
            tempArr1[27:39], tempArr2[27:39] = tempArr2[27:39], tempArr1[27:39]
        else:
            tempArr1[26:39], tempArr2[26:39] = tempArr2[26:39], tempArr1[26:39]     
    elif ((x >= 104) and (x <= 111)):
        if(x != 111):
            tempArr1, tempArr2 = genSwitchMask(x-104, 25, tempArr1, tempArr2)
            tempArr1[26:39], tempArr2[26:39] = tempArr2[26:39], tempArr1[26:39]
        else:
            tempArr1[25:39], tempArr2[25:39] = tempArr2[25:39], tempArr1[25:39]  
    elif ((x >= 112) and (x <= 119)):
        if(x != 119):
            tempArr1, tempArr2 = genSwitchMask(x-112, 24, tempArr1, tempArr2)
            tempArr1[25:39], tempArr2[25:39] = tempArr2[25:39], tempArr1[25:39]
        else:
            tempArr1[24:39], tempArr2[24:39] = tempArr2[24:39], tempArr1[24:39]            
    elif ((x >= 120) and (x <= 127)):
        if(x != 127):
            tempArr1, tempArr2 = genSwitchMask(x-120, 23, tempArr1, tempArr2)
            tempArr1[24:39], tempArr2[24:39] = tempArr2[24:39], tempArr1[24:39]
        else:
            tempArr1[23:39], tempArr2[23:39] = tempArr2[23:39], tempArr1[23:39] 
    elif ((x >= 128) and (x <= 135)):
        if(x != 135):
            tempArr1, tempArr2 = genSwitchMask(x-128, 22, tempArr1, tempArr2)
            tempArr1[23:39], tempArr2[23:39] = tempArr2[23:39], tempArr1[23:39]
        else:
            tempArr1[22:39], tempArr2[22:39] = tempArr2[22:39], tempArr1[22:39] 
    elif ((x >= 136) and (x <= 143)):
        if(x != 143):
            tempArr1, tempArr2 = genSwitchMask(x-136, 21, tempArr1, tempArr2)
            tempArr1[22:39], tempArr2[22:39] = tempArr2[22:39], tempArr1[22:39]
        else:
            tempArr1[21:39], tempArr2[21:39] = tempArr2[21:39], tempArr1[21:39]           
    elif ((x >= 144) and (x <= 151)):
        if(x != 151):
            tempArr1, tempArr2 = genSwitchMask(x-144, 20, tempArr1, tempArr2)
            tempArr1[21:39], tempArr2[21:39] = tempArr2[21:39], tempArr1[21:39]
        else:
            tempArr1[20:39], tempArr2[20:39] = tempArr2[20:39], tempArr1[20:39]         
    elif ((x >= 152) and (x <= 159)):
        if(x != 159):
            tempArr1, tempArr2 = genSwitchMask(x-152, 19, tempArr1, tempArr2)
            tempArr1[20:39], tempArr2[20:39] = tempArr2[20:39], tempArr1[20:39]
        else:
            tempArr1[19:39], tempArr2[19:39] = tempArr2[19:39], tempArr1[19:39] 
    elif ((x >= 160) and (x <= 167)):
        if(x != 167):
            tempArr1, tempArr2 = genSwitchMask(x-160, 18, tempArr1, tempArr2)
            tempArr1[19:39], tempArr2[19:39] = tempArr2[19:39], tempArr1[19:39]
        else:
            tempArr1[18:39], tempArr2[18:39] = tempArr2[18:39], tempArr1[18:39]       
    elif ((x >= 168) and (x <= 175)):
        if(x != 175):
            tempArr1, tempArr2 = genSwitchMask(x-168, 17, tempArr1, tempArr2)
            tempArr1[18:39], tempArr2[18:39] = tempArr2[18:39], tempArr1[18:39]
        else:
            tempArr1[17:39], tempArr2[17:39] = tempArr2[17:39], tempArr1[17:39] 
    elif ((x >= 176) and (x <= 183)):
        if(x != 183):
            tempArr1, tempArr2 = genSwitchMask(x-176, 16, tempArr1, tempArr2)
            tempArr1[17:39], tempArr2[17:39] = tempArr2[17:39], tempArr1[17:39]
        else:
            tempArr1[16:39], tempArr2[16:39] = tempArr2[16:39], tempArr1[16:39] 
    elif ((x >= 184) and (x <= 191)):
        if(x != 191):
            tempArr1, tempArr2 = genSwitchMask(x-184, 15, tempArr1, tempArr2)
            tempArr1[16:39], tempArr2[16:39] = tempArr2[16:39], tempArr1[16:39]
        else:
            tempArr1[15:39], tempArr2[15:39] = tempArr2[15:39], tempArr1[15:39] 
    elif ((x >= 192) and (x <= 199)):
        if(x != 199):
            tempArr1, tempArr2 = genSwitchMask(x-192, 14, tempArr1, tempArr2)
            tempArr1[15:39], tempArr2[15:39] = tempArr2[15:39], tempArr1[15:39]
        else:
            tempArr1[14:39], tempArr2[14:39] = tempArr2[14:39], tempArr1[14:39]             
    elif ((x >= 200) and (x <= 207)):
        if(x != 207):
            tempArr1, tempArr2 = genSwitchMask(x-200, 13, tempArr1, tempArr2)
            tempArr1[14:39], tempArr2[14:39] = tempArr2[14:39], tempArr1[14:39]
        else:
            tempArr1[13:39], tempArr2[13:39] = tempArr2[13:39], tempArr1[13:39] 
    elif ((x >= 208) and (x <= 215)):
        if(x != 215):
            tempArr1, tempArr2 = genSwitchMask(x-208, 12, tempArr1, tempArr2)
            tempArr1[13:39], tempArr2[13:39] = tempArr2[13:39], tempArr1[13:39]
        else:
            tempArr1[12:39], tempArr2[12:39] = tempArr2[12:39], tempArr1[12:39]                
    elif ((x >= 216) and (x <= 223)):
        if(x != 223):
            tempArr1, tempArr2 = genSwitchMask(x-216, 11, tempArr1, tempArr2)
            tempArr1[12:39], tempArr2[12:39] = tempArr2[12:39], tempArr1[12:39]
        else:
            tempArr1[11:39], tempArr2[11:39] = tempArr2[11:39], tempArr1[11:39]                    
    elif ((x >= 224) and (x <= 231)):
        if(x != 231):
            tempArr1, tempArr2 = genSwitchMask(x-224, 10, tempArr1, tempArr2)
            tempArr1[11:39], tempArr2[11:39] = tempArr2[11:39], tempArr1[11:39]
        else:
            tempArr1[10:39], tempArr2[10:39] = tempArr2[10:39], tempArr1[10:39]       
    elif ((x >= 232) and (x <= 239)):
        if(x != 239):
            tempArr1, tempArr2 = genSwitchMask(x-232, 9, tempArr1, tempArr2)
            tempArr1[10:39], tempArr2[10:39] = tempArr2[10:39], tempArr1[10:39]
        else:
            tempArr1[9:39], tempArr2[9:39] = tempArr2[9:39], tempArr1[9:39]     
    elif ((x >= 240) and (x <= 247)):
        if(x != 247):
            tempArr1, tempArr2 = genSwitchMask(x-240, 8, tempArr1, tempArr2)
            tempArr1[9:39], tempArr2[9:39] = tempArr2[9:39], tempArr1[9:39]
        else:
            tempArr1[8:39], tempArr2[8:39] = tempArr2[8:39], tempArr1[8:39]     
    elif ((x >= 248) and (x <= 255)):
        if(x != 255):
            tempArr1, tempArr2 = genSwitchMask(x-248, 7, tempArr1, tempArr2)
            tempArr1[8:39], tempArr2[8:39] = tempArr2[8:39], tempArr1[8:39]
        else:
            tempArr1[7:39], tempArr2[7:39] = tempArr2[7:39], tempArr1[7:39] 
    elif ((x >= 256) and (x <= 263)):
        if(x != 263):
            tempArr1, tempArr2 = genSwitchMask(x-256, 6, tempArr1, tempArr2)
            tempArr1[7:39], tempArr2[7:39] = tempArr2[7:39], tempArr1[7:39]
        else:
            tempArr1[6:39], tempArr2[6:39] = tempArr2[6:39], tempArr1[6:39] 
    elif ((x >= 264) and (x <= 271)):
        if(x != 271):
            tempArr1, tempArr2 = genSwitchMask(x-264, 5, tempArr1, tempArr2)
            tempArr1[6:39], tempArr2[6:39] = tempArr2[6:39], tempArr1[6:39]
        else:
            tempArr1[5:39], tempArr2[5:39] = tempArr2[5:39], tempArr1[5:39] 
    elif ((x >= 272) and (x <= 279)):
        if(x != 279):
            tempArr1, tempArr2 = genSwitchMask(x-272, 4, tempArr1, tempArr2)
            tempArr1[5:39], tempArr2[5:39] = tempArr2[5:39], tempArr1[5:39]
        else:
            tempArr1[4:39], tempArr2[4:39] = tempArr2[4:39], tempArr1[4:39]
    elif ((x >= 280) and (x <= 287)):
        if(x != 287):
            tempArr1, tempArr2 = genSwitchMask(x-280, 3, tempArr1, tempArr2)
            tempArr1[4:39], tempArr2[4:39] = tempArr2[4:39], tempArr1[4:39]
        else:
            tempArr1[3:39], tempArr2[3:39] = tempArr2[3:39], tempArr1[3:39]
    elif ((x >= 288) and (x <= 295)):
        if(x != 295):
            tempArr1, tempArr2 = genSwitchMask(x-288, 2, tempArr1, tempArr2)
            tempArr1[3:39], tempArr2[3:39] = tempArr2[3:39], tempArr1[3:39]
        else:
            tempArr1[2:39], tempArr2[2:39] = tempArr2[2:39], tempArr1[2:39]            
    elif ((x >= 296) and (x <= 303)):
        if(x != 303):
            tempArr1, tempArr2 = genSwitchMask(x-296, 1, tempArr1, tempArr2)
            tempArr1[2:39], tempArr2[2:39] = tempArr2[2:39], tempArr1[2:39]
        else:
            tempArr1[1:39], tempArr2[1:39] = tempArr2[1:39], tempArr1[1:39]   
    elif ((x >= 304) and (x <= 310)):
        tempArr1, tempArr2 = genSwitchMask(x-304, 0, tempArr1, tempArr2)
        tempArr1[1:39], tempArr2[1:39] = tempArr2[1:39], tempArr1[1:39]
    
    return tempArr1, tempArr2


def genMutation(arr1):
    val = np.uint8(0)

    for i in range(0, MUTATION_RATE):
        y = int(random.random() * POPSIZE)
        
        for j in range(0, DOUBLE_MUTATION):
            x = int(random.random() * 312) 
        
            if ((x >= 0) and (x <= 7)):
                val = np.uint8(arr1[y][38] & (1 << x))
                if(False != val):
                    arr1[y][38] = arr1[y][38] & np.uint8(~val)
                else:
                    val = (1 << x)
                    arr1[y][38] = arr1[y][38] | val
            elif ((x >= 8) and (x <= 15)):
                val = np.uint8(arr1[y][37] & (1 << x-8))
                if(False != val):
                    arr1[y][37] = arr1[y][37] & np.uint8(~val)
                else:
                    val = (1 << x-8)
                    arr1[y][37] = arr1[y][37] | val  
            elif ((x >= 16) and (x <= 23)):
                val = np.uint8(arr1[y][36] & (1 << x-16))
                if(False != val):
                    arr1[y][36] = arr1[y][36] & np.uint8(~val)
                else:
                    val = (1 << x-16)
                    arr1[y][36] = arr1[y][36] | val  
            elif ((x >= 24) and (x <= 31)):
                val = np.uint8(arr1[y][35] & (1 << x-24))
                if(False != val):
                    arr1[y][35] = arr1[y][35] & np.uint8(~val)
                else:
                    val = (1 << x-24)
                    arr1[y][35] = arr1[y][35] | val     
            elif ((x >= 32) and (x <= 39)):
                val = np.uint8(arr1[y][34] & (1 << x-32))
                if(False != val):
                    arr1[y][34] = arr1[y][34] & np.uint8(~val)
                else:
                    val = (1 << x-32)
                    arr1[y][34] = arr1[y][34] | val 
            elif ((x >= 40) and (x <= 47)):
                val = np.uint8(arr1[y][33] & (1 << x-40))
                if(False != val):
                    arr1[y][33] = arr1[y][33] & np.uint8(~val)
                else:
                    val = (1 << x-40)
                    arr1[y][33] = arr1[y][33] | val 
            elif ((x >= 48) and (x <= 55)):
                val = np.uint8(arr1[y][32] & (1 << x-48))
                if(False != val):
                    arr1[y][32] = arr1[y][32] & np.uint8(~val)
                else:
                    val = (1 << x-48)
                    arr1[y][32] = arr1[y][32] | val 
            elif ((x >= 56) and (x <= 63)):
                val = np.uint8(arr1[y][31] & (1 << x-56))
                if(False != val):
                    arr1[y][31] = arr1[y][31] & np.uint8(~val)
                else:
                    val = (1 << x-56)
                    arr1[y][31] = arr1[y][31] | val 
            elif ((x >= 64) and (x <= 71)):
                val = np.uint8(arr1[y][30] & (1 << x-64))
                if(False != val):
                    arr1[y][30] = arr1[y][30] & np.uint8(~val)
                else:
                    val = (1 << x-64)
                    arr1[y][30] = arr1[y][30] | val 
            elif ((x >= 72) and (x <= 79)):
                val = np.uint8(arr1[y][29] & (1 << x-72))
                if(False != val):
                    arr1[y][29] = arr1[y][29] & np.uint8(~val)
                else:
                    val = (1 << x-72)
                    arr1[y][29] = arr1[y][29] | val 
            elif ((x >= 80) and (x <= 87)):
                val = np.uint8(arr1[y][28] & (1 << x-80))
                if(False != val):
                    arr1[y][28] = arr1[y][28] & np.uint8(~val)
                else:
                    val = (1 << x-80)
                    arr1[y][28] = arr1[y][28] | val  
            elif ((x >= 88) and (x <= 95)):
                val = np.uint8(arr1[y][27] & (1 << x-88))
                if(False != val):
                    arr1[y][27] = arr1[y][27] & np.uint8(~val)
                else:
                    val = (1 << x-88)
                    arr1[y][27] = arr1[y][27] | val 
            elif ((x >= 96) and (x <= 103)):
                val = np.uint8(arr1[y][26] & (1 << x-96))
                if(False != val):
                    arr1[y][26] = arr1[y][26] & np.uint8(~val)
                else:
                    val = (1 << x-96)
                    arr1[y][26] = arr1[y][26] | val 
            elif ((x >= 104) and (x <= 111)):
                val = np.uint8(arr1[y][25] & (1 << x-104))
                if(False != val):
                    arr1[y][25] = arr1[y][25] & np.uint8(~val)
                else:
                    val = (1 << x-104)
                    arr1[y][25] = arr1[y][25] | val
            elif ((x >= 112) and (x <= 119)):
                val = np.uint8(arr1[y][24] & (1 << x-112))
                if(False != val):
                    arr1[y][24] = arr1[y][24] & np.uint8(~val)
                else:
                    val = (1 << x-112)
                    arr1[y][24] = arr1[y][24] | val
            elif ((x >= 120) and (x <= 127)):
                val = np.uint8(arr1[y][23] & (1 << x-120))
                if(False != val):
                    arr1[y][23] = arr1[y][23] & np.uint8(~val)
                else:
                    val = (1 << x-120)
                    arr1[y][23] = arr1[y][23] | val
            elif ((x >= 128) and (x <= 135)):
                val = np.uint8(arr1[y][22] & (1 << x-128))
                if(False != val):
                    arr1[y][22] = arr1[y][22] & np.uint8(~val)
                else:
                    val = (1 << x-128)
                    arr1[y][22] = arr1[y][22] | val
            elif ((x >= 136) and (x <= 143)):
                val = np.uint8(arr1[y][21] & (1 << x-136))
                if(False != val):
                    arr1[y][21] = arr1[y][21] & np.uint8(~val)
                else:
                    val = (1 << x-136)
                    arr1[y][21] = arr1[y][21] | val
            elif ((x >= 144) and (x <= 151)):
                val = np.uint8(arr1[y][20] & (1 << x-144))
                if(False != val):
                    arr1[y][20] = arr1[y][20] & np.uint8(~val)
                else:
                    val = (1 << x-144)
                    arr1[y][20] = arr1[y][20] | val
            elif ((x >= 152) and (x <= 159)):
                val = np.uint8(arr1[y][19] & (1 << x-152))
                if(False != val):
                    arr1[y][19] = arr1[y][19] & np.uint8(~val)
                else:
                    val = (1 << x-152)
                    arr1[y][19] = arr1[y][19] | val
            elif ((x >= 160) and (x <= 167)):
                val = np.uint8(arr1[y][18] & (1 << x-160))
                if(False != val):
                    arr1[y][18] = arr1[y][18] & np.uint8(~val)
                else:
                    val = (1 << x-160)
                    arr1[y][18] = arr1[y][18] | val                  
            elif ((x >= 168) and (x <= 175)):
                val = np.uint8(arr1[y][17] & (1 << x-168))
                if(False != val):
                    arr1[y][17] = arr1[y][17] & np.uint8(~val)
                else:
                    val = (1 << x-168)
                    arr1[y][17] = arr1[y][17] | val                                 
            elif ((x >= 176) and (x <= 183)):
                val = np.uint8(arr1[y][16] & (1 << x-176))
                if(False != val):
                    arr1[y][16] = arr1[y][16] & np.uint8(~val)
                else:
                    val = (1 << x-176)
                    arr1[y][16] = arr1[y][16] | val                    
            elif ((x >= 184) and (x <= 191)):
                val = np.uint8(arr1[y][15] & (1 << x-184))
                if(False != val):
                    arr1[y][15] = arr1[y][15] & np.uint8(~val)
                else:
                    val = (1 << x-184)
                    arr1[y][15] = arr1[y][15] | val                             
            elif ((x >= 192) and (x <= 199)):
                val = np.uint8(arr1[y][14] & (1 << x-192))
                if(False != val):
                    arr1[y][14] = arr1[y][14] & np.uint8(~val)
                else:
                    val = (1 << x-192)
                    arr1[y][14] = arr1[y][14] | val   
            elif ((x >= 200) and (x <= 207)):
                val = np.uint8(arr1[y][13] & (1 << x-200))
                if(False != val):
                    arr1[y][13] = arr1[y][13] & np.uint8(~val)
                else:
                    val = (1 << x-200)
                    arr1[y][13] = arr1[y][13] | val   
            elif ((x >= 208) and (x <= 215)):
                val = np.uint8(arr1[y][12] & (1 << x-208))
                if(False != val):
                    arr1[y][12] = arr1[y][12] & np.uint8(~val)
                else:
                    val = (1 << x-208)
                    arr1[y][12] = arr1[y][12] | val 
            elif ((x >= 216) and (x <= 223)):
                val = np.uint8(arr1[y][11] & (1 << x-216))
                if(False != val):
                    arr1[y][11] = arr1[y][11] & np.uint8(~val)
                else:
                    val = (1 << x-216)
                    arr1[y][11] = arr1[y][11] | val 
            elif ((x >= 224) and (x <= 231)):
                val = np.uint8(arr1[y][10] & (1 << x-224))
                if(False != val):
                    arr1[y][10] = arr1[y][10] & np.uint8(~val)
                else:
                    val = (1 << x-224)
                    arr1[y][10] = arr1[y][10] | val                
            elif ((x >= 232) and (x <= 239)):
                val = np.uint8(arr1[y][9] & (1 << x-232))
                if(False != val):
                    arr1[y][9] = arr1[y][9] & np.uint8(~val)
                else:
                    val = (1 << x-232)
                    arr1[y][9] = arr1[y][9] | val                 
            elif ((x >= 240) and (x <= 247)):
                val = np.uint8(arr1[y][8] & (1 << x-240))
                if(False != val):
                    arr1[y][8] = arr1[y][8] & np.uint8(~val)
                else:
                    val = (1 << x-240)
                    arr1[y][8] = arr1[y][8] | val   
            elif ((x >= 248) and (x <= 255)):
                val = np.uint8(arr1[y][7] & (1 << x-248))
                if(False != val):
                    arr1[y][7] = arr1[y][7] & np.uint8(~val)
                else:
                    val = (1 << x-248)
                    arr1[y][7] = arr1[y][7] | val      
            elif ((x >= 256) and (x <= 263)):
                val = np.uint8(arr1[y][6] & (1 << x-256))
                if(False != val):
                    arr1[y][6] = arr1[y][6] & np.uint8(~val)
                else:
                    val = (1 << x-256)
                    arr1[y][6] = arr1[y][6] | val  
            elif ((x >= 264) and (x <= 271)):
                val = np.uint8(arr1[y][5] & (1 << x-264))
                if(False != val):
                    arr1[y][5] = arr1[y][5] & np.uint8(~val)
                else:
                    val = (1 << x-264)
                    arr1[y][5] = arr1[y][5] | val 
            elif ((x >= 272) and (x <= 279)):
                val = np.uint8(arr1[y][4] & (1 << x-272))
                if(False != val):
                    arr1[y][4] = arr1[y][4] & np.uint8(~val)
                else:
                    val = (1 << x-272)
                    arr1[y][4] = arr1[y][4] | val 
            elif ((x >= 280) and (x <= 287)):
                val = np.uint8(arr1[y][3] & (1 << x-280))
                if(False != val):
                    arr1[y][3] = arr1[y][3] & np.uint8(~val)
                else:
                    val = (1 << x-280)
                    arr1[y][3] = arr1[y][3] | val 
            elif ((x >= 288) and (x <= 295)):
                val = np.uint8(arr1[y][2] & (1 << x-288))
                if(False != val):
                    arr1[y][2] = arr1[y][2] & np.uint8(~val)
                else:
                    val = (1 << x-288)
                    arr1[y][2] = arr1[y][2] | val 
            elif ((x >= 296) and (x <= 303)):
                val = np.uint8(arr1[y][1] & (1 << x-296))
                if(False != val):
                    arr1[y][1] = arr1[y][1] & np.uint8(~val)
                else:
                    val = (1 << x-296)
                    arr1[y][1] = arr1[y][1] | val                    
            elif ((x >= 304) and (x <= 311)):
                val = np.uint8(arr1[y][0] & (1 << x-304))
                if(False != val):
                    arr1[y][0] = arr1[y][0] & np.uint8(~val)
                else:
                    val = (1 << x-304)
                    arr1[y][0] = arr1[y][0] | val                   
  



              

def nextGeneration(popLastGen):
    childPop = []
    
    for x in range(0, int(POPSIZE/2)):
        popRand = []
        for y in range(2):
            localLst = []
            for z in range(int(CHILD_SIZE)):
                localRand = int(random.random() * POPSIZE)
                localLst.append(popLastGen[localRand])
            localLstOrdered = sorted(localLst, key=operator.itemgetter(39))            
            popRand.append(localLstOrdered[0])
        popRand[0], popRand[1] = genSwitch(popRand[0], popRand[1])
        childPop.append(popRand[0])
        childPop.append(popRand[1])
    genMutation(childPop)
    getCurveError(childPop)

    for w in range(0, POPSIZE):
        popLastGen.append(childPop[w])
    popLastGen = sorted(popLastGen, key=operator.itemgetter(39))      
    popLastGen = popLastGen[:POPSIZE]

    
    return popLastGen


def printPlot(localPop, gen):
    local = 0
    z, mf1, mf2, mf3, mf4, mf5, mf6 = [], [], [], [], [], [], []
    print("\nGeneration: " + str(gen) + " - Error: " + str(localPop[0][39]))
    
    
    for i in range(0,15):
        z.append([])
        for j in range(0,12):

            mf1.append(math.exp((-(X[i]-(localPop[0][0]/I0))**2)/(2*(localPop[0][1]/I1)**2)))
            mf2.append(math.exp((-(X[i]-(localPop[0][2]/I2))**2)/(2*(localPop[0][3]/I3)**2)))
            mf3.append(math.exp((-(X[i]-(localPop[0][4]/I4))**2)/(2*(localPop[0][5]/I5)**2)))
            mf4.append(math.exp((-(Y[j]-(localPop[0][6]/I6))**2)/(2*(localPop[0][7]/I7)**2)))
            mf5.append(math.exp((-(Y[j]-(localPop[0][8]/I8))**2)/(2*(localPop[0][9]/I9)**2)))
            mf6.append(math.exp((-(Y[j]-(localPop[0][10]/I10))**2)/(2*(localPop[0][11]/I11)**2)))
            
            inf1=mf1[i]*mf4[j]
            inf2=mf1[i]*mf5[j]
            inf3=mf1[i]*mf6[j]
            inf4=mf2[i]*mf4[j]
            inf5=mf2[i]*mf5[j]
            inf6=mf2[i]*mf6[j]
            inf7=mf3[i]*mf4[j]
            inf8=mf3[i]*mf5[j]
            inf9=mf3[i]*mf6[j]
        
            p1 = (localPop[0][12]/I12)-19
            q3 = (localPop[0][19]/I19)-2
    
            reg1=inf1*(p1*X[i]+(localPop[0][13]/I13)*Y[j]+(localPop[0][14]/I14))
            reg2=inf2*((localPop[0][15]/I15)*X[i]+(localPop[0][16]/I16)*Y[j]+(localPop[0][17]/I17))
            reg3=inf1*((localPop[0][18]/I18)*X[i]+q3*Y[j]+(localPop[0][20]/I20))
            reg4=inf1*((localPop[0][21]/I21)*X[i]+(localPop[0][22]/I22)*Y[j]+(localPop[0][23]/I23))
            reg5=inf1*((localPop[0][24]/I24)*X[i]+(localPop[0][25]/I25)*Y[j]+(localPop[0][26]/I26))
            reg6=inf1*((localPop[0][27]/I27)*X[i]+(localPop[0][28]/I28)*Y[j]+(localPop[0][29]/I29))
            reg7=inf1*((localPop[0][30]/I30)*X[i]+(localPop[0][31]/I31)*Y[j]+(localPop[0][32]/I32))
            reg8=inf1*((localPop[0][33]/I33)*X[i]+(localPop[0][34]/I34)*Y[j]+(localPop[0][35]/I35))
            reg9=inf1*((localPop[0][36]/I36)*X[i]+(localPop[0][37]/I37)*Y[j]+(localPop[0][38]/I38))
    
            b=inf1+inf2+inf3+inf4+inf5+inf6+inf7+inf8+inf9
            total=reg1+reg2+reg3+reg4+reg5+reg6+reg7+reg8+reg9
            
            z[i].append(total/b)

            if((Z[i][j] - z[i][j]) > 1.5):
                z[i][j] += 1.5
            elif((Z[i][j] - z[i][j]) > 1):
                z[i][j] += 1
            elif((Z[i][j] - z[i][j]) > 0.5):
                z[i][j] += 0.5
            elif((Z[i][j] - z[i][j]) < -1.5):
                z[i][j] -= 1.5
            elif((Z[i][j] - z[i][j]) < -1):
                z[i][j] -= 1
            elif((Z[i][j] - z[i][j]) < -0.5):
                z[i][j] -= 0.5
                    
                    
    progress.append(localPop[0][39])
    
    
    ax.set_xlabel('Meses')
    ax.set_ylabel('Colonias')
    ax.set_zlabel('Precio')
    surf = ax.plot_surface(Xp, Yp, np.array(z), cmap=cm.Greens, linewidth=0)
    surf2 = ax.plot_surface(Xp, Yp, Z, cmap=cm.PuBu, linewidth=0)

    ax2.plot(progress)
    ax2.title.set_text('Error')
    
    ax3.plot(Xfuzzyp, mf1, label = 'linear', color = "red")
    ax3.plot(Xfuzzyp, mf2, label = 'linear', color = "blue")
    ax3.plot(Xfuzzyp, mf3, label = 'linear', color = "green")
    ax3.title.set_text('FuzzySets')
    
    #ax4.plot(Xfuzzyp, mf4, label = 'linear', color = "red")
    #ax4.plot(Xfuzzyp, mf5, label = 'linear', color = "blue")
    #ax4.plot(Xfuzzyp, mf6, label = 'linear', color = "green")
    #ax4.title.set_text('FuzzySets')
    
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    ax.clear()
    ax2.clear()
    ax3.clear()
    
            

def main():
    population = setInitPopulation()
    print(population[0])
    #for x in range(POPSIZE/2):
    #population[0] = [126, 77, 183, 169, 58, 29, 38, 117, 142, 111, 53, 231, 170, 97, 63, 227, 41, 9, 243, 19, 212, 175, 11, 231, 203, 11, 248, 10, 163, 140, 139, 32, 65, 143, 59, 113, 78, 16, 222, 58813.41055632782]
    getCurveError(population)
    
    fullPop = [ x for x in population[:]]
    for i in range(100):
        fullPop = nextGeneration(fullPop)
        printPlot(fullPop, i)
    plt.pause(100)
    print(fullPop[0])
    
        
    #localLstOrdered = sorted(population, key=operator.itemgetter(39))
    #print(localLstOrdered[0][39])

if __name__ == "__main__":
   main()