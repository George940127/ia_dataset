
from asyncio.windows_events import NULL
from cProfile import label
import random
import math
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import cm



class SurfaceAdjustment:
    
    plotResolution = 40
    TotalSuburb = 11
    fig = 0
    ax1 = 0
    ax1 = 0
    ax2 = 0
    #peso =round(255 / 45) 
    peso = 5
    mutation = False
    elitism = False
    mutationPercentage = 0

    def __init__(self, mutationPercentage = 0, elitism = False):
        random.seed()
        if mutationPercentage == 0:
            self.mutation = False
        else:
            self.mutation = True
            self.mutationPercentage = mutationPercentage 
        self.elitism = elitism

    def createZ_ReferenceDataFrame(self):
        "Convert csv data file in a Data Frame through Pandas Library"
        dataFrame = pd.read_csv("C:\\Users\\roxy_\\Documents\\MCC\\2022-1\\Inteligencia Artificial\\Proyecto\\DatosParaGraficarUnPlano.csv", index_col = "id")
        #print(f"dataFrame = {dataFrame}")
        float_DF = dataFrame.astype(dtype='float', copy = True, errors='raise')
        return float_DF

    def Create_TakagiSugenoSurface(self,parametersList): #parametersList = [m1,m2,m3,m4,m5,m6,d1,d2,d3,d4,d5,d6,p1,p2,p3,p4,p5,p6,p7,p8,p9,q1,q2,q3,q4,q5,q6,q7,q8,q9,r1,r2,r3,r4,r5,r6,r7,r8,r9]
        #self.plotResolution = 11
        m1 = parametersList[0]
        m2 = parametersList[1]
        m3 = parametersList[2]
        m4 = parametersList[3]
        m5 = parametersList[4]
        m6 = parametersList[5]
        for i in range(6,12):
            if parametersList[i] == 0:
                parametersList[i] = parametersList[i] + 1
        d1 = parametersList[6]
        d2 = parametersList[7]
        d3 = parametersList[8]
        d4 = parametersList[9]
        d5 = parametersList[10]
        d6 = parametersList[11]
        p1 = parametersList[12]
        p2 = parametersList[13]
        p3 = parametersList[14]
        p4 = parametersList[15]
        p5 = parametersList[16]
        p6 = parametersList[17]
        p7 = parametersList[18]
        p8 = parametersList[19]
        p9 = parametersList[20]
        q1 = parametersList[21]
        q2 = parametersList[22]
        q3 = parametersList[23]
        q4 = parametersList[24]
        q5 = parametersList[25]
        q6 = parametersList[26]
        q7 = parametersList[27]
        q8 = parametersList[28]
        q9 = parametersList[29]
        r1 = parametersList[30]
        r2 = parametersList[31]
        r3 = parametersList[32]
        r4 = parametersList[33]
        r5 = parametersList[34]
        r6 = parametersList[35]
        r7 = parametersList[36]
        r8 = parametersList[37]
        r9 = parametersList[38]

        mf1 = []
        mf2 = []
        mf3 = []
        mf4 = []
        mf5 = []
        mf6 = []
        inf1 = [[0] * 11  for i in range(11)]
        inf2 = [[0] * 11  for i in range(11)]
        inf3 = [[0] * 11  for i in range(11)]
        inf4 = [[0] * 11  for i in range(11)]
        inf5 = [[0] * 11  for i in range(11)]
        inf6 = [[0] * 11  for i in range(11)]
        inf7 = [[0] * 11  for i in range(11)]
        inf8 = [[0] * 11  for i in range(11)]
        inf9 = [[0] * 11  for i in range(11)]
        reg1 = [[0] * 11  for i in range(11)]
        reg2 = [[0] * 11  for i in range(11)]
        reg3 = [[0] * 11  for i in range(11)]
        reg4 = [[0] * 11  for i in range(11)]
        reg5 = [[0] * 11  for i in range(11)]
        reg6 = [[0] * 11  for i in range(11)]
        reg7 = [[0] * 11  for i in range(11)]
        reg8 = [[0] * 11  for i in range(11)]
        reg9 = [[0] * 11  for i in range(11)]
        a = [[0] * 11  for i in range(11)]
        b = [[0] * 11  for i in range(11)]
        z = [[0] * 11  for i in range(11)]

        x = []
        y = []
        # for n in range(1,self.plotResolution+1):
        #     x.append(n/self.plotResolution)
        #     y.append(n/self.plotResolution)

        for i in range(0,11):
            for j in range(0,11):
                x.append(i/11)
                y.append(j/11)

                #Parameterized Gaussian Membership Functions
                mf1.append(pow(math.e,-0.5*pow((x[i]-m1)/d1,2)))
                mf2.append(pow(math.e,-0.5*pow((x[i]-m2)/d2,2)))
                mf3.append(pow(math.e,-0.5*pow((x[i]-m3)/d3,2)))

                mf4.append(pow(math.e,-0.5*pow((y[j]-m4)/d4,2)))
                mf5.append(pow(math.e,-0.5*pow((y[j]-m5)/d5,2)))
                mf6.append(pow(math.e,-0.5*pow((y[j]-m6)/d6,2)))

                inf1[i][j] = mf1[i]*mf4[j]
                inf2[i][j] = mf1[i]*mf5[j]
                inf3[i][j] = mf1[i]*mf6[j]

                inf4[i][j] = mf2[i]*mf4[j]
                inf5[i][j] = mf2[i]*mf5[j]
                inf6[i][j] = mf2[i]*mf6[j]

                inf7[i][j] = mf3[i]*mf4[j]
                inf8[i][j] = mf3[i]*mf5[j]
                inf9[i][j] = mf3[i]*mf6[j]

                reg1[i][j] = inf1[i][j]*((p1*x[i])+(q1*y[j])+r1)
                reg2[i][j] = inf2[i][j]*((p2*x[i])+(q2*y[j])+r2)
                reg3[i][j] = inf3[i][j]*((p3*x[i])+(q3*y[j])+r3)
                
                reg4[i][j] = inf4[i][j]*((p4*x[i])+(q4*y[j])+r4)
                reg5[i][j] = inf5[i][j]*((p5*x[i])+(q5*y[j])+r5)
                reg6[i][j] = inf6[i][j]*((p6*x[i])+(q6*y[j])+r6)
                
                reg7[i][j] = inf7[i][j]*((p7*x[i])+(q7*y[j])+r7)
                reg8[i][j] = inf7[i][j]*((p8*x[i])+(q8*y[j])+r8)
                reg9[i][j] = inf7[i][j]*((p9*x[i])+(q9*y[j])+r9)

                b[i][j] = inf1[i][j] + inf2[i][j] + inf3[i][j] + inf4[i][j] + inf5[i][j] + inf6[i][j] + inf7[i][j] + inf8[i][j] + inf9[i][j]
                a[i][j] = reg1[i][j] + reg2[i][j] + reg3[i][j] + reg4[i][j] + reg5[i][j] + reg6[i][j] + reg7[i][j] + reg8[i][j] + reg9[i][j]
                z[i][j] = a[i][j] / b[i][j]

        Z = pd.DataFrame(data=z)
        float_Z = Z.astype(dtype='float', copy = True, errors='raise')
        float_Z.columns = ['cero', 'uno', 'dos', 'tres','cuatro', 'cinco','seis','siete','ocho','nueve', 'diez']
        float_Z.index = ['id_0','id_1','id_2','id_3','id_4','id_5','id_6','id_7','id_8','id_9','id_10']
        #print(f"Result TS --> Z = {float_Z}")
        return float_Z

    def drawGraphs(self,z1_RefValues, chromosome, x2_Values, y2_Values, generation, population):
        """Plot surfaces, aptitude function and fuzzy sets"""
        plt.ion() # Turn on interactive mode using
        #self.fig.canvas.draw()
        #self.fig.canvas.flush_events()
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        if self.mutation == True and self.elitism == False:
            plt.suptitle("Generación: " + str(generation+1) + ";" + " Población: " + str(population) + " individuos" + ";" + " Mutación: " + str(self.mutationPercentage)+ "% de la población" + ";" +" Elitismo:" + " No habilitado")
        elif self.mutation == False and self.elitism == True:
            plt.suptitle("Generación: " + str(generation+1) + ";" + " Población: " + str(population) + " individuos" + ";" +" Mutación: " + "No habilitada " + ";" + " Elitismo:" + " Habilitado")
        elif self.mutation == True and self.elitism == True:
            plt.suptitle("Generación: " + str(generation+1) + ";" + " Población: " + str(population) + " individuos" + ";" + " Mutación: " + str(self.mutationPercentage) + "% de la población" + ";" + " Elitismo:" + " Habilitado")
        else:
            plt.suptitle("Generación: " + str(generation+1) + ";" + " Población: " + str(population) + " individuos" + ";" + " Mutación: " + "No habilitada " + ";" + " Elitismo:" + "No habilitado")

        self.ax1.set_xlabel('Fecha')
        self.ax1.set_ylabel('Colonia')
        self.ax1.set_zlabel('Precio del inmueble')
       # X = np.arange(1/self.plotResolution, 1 + 1/self.plotResolution, 1/self.plotResolution)
       # Y = np.arange(1/self.plotResolution, 1 + 1/self.plotResolution, 1/self.plotResolution)
        X = np.arange(0,11,1)
        Y = np.arange(0,11,1)
        X, Y = np.meshgrid(X, Y)
        refZ = np.array(z1_RefValues) #Convert a Data Frame in an array
        z1_CalculatedValues = self.Create_TakagiSugenoSurface(chromosome)
        calculatedZ = np.array(z1_CalculatedValues)
        surf = self.ax1.plot_surface(X, Y, refZ, cmap= cm.coolwarm, linewidth=0)
        surf2 = self.ax1.plot_surface(X, Y, calculatedZ, cmap=cm.seismic, linewidth=0)

        # Draw Aptitude Function
        #plt.title(" Integral del error entre curvas ")
        self.ax2.grid()
        self.ax2.set_xlabel('Generación')
        self.ax2.set_ylabel('Función de Aptitud')
        self.ax2.plot(x2_Values, y2_Values, color='tab:orange',  marker='*')
        self.ax2.text(x2_Values[len(x2_Values)-1], y2_Values[len(y2_Values)-1],"f(x)= " + str(round(y2_Values[len(y2_Values)-1],2)))

        #Draw Membership Functions
        self.ax3.title.set_text('Conjuntos Difusos')
        self.ax3.grid()
        self.ax3.set_xlabel('Conjuntos Difusos: X -> Fecha Y-> Colonia')
        self.ax3.set_ylabel('Grados de pertenencia')
        mf1 = []
        mf2 = []
        mf3 = []
        mf4 = []
        mf5 = []
        mf6 = []
        m1 = chromosome[0]
        m2 = chromosome[1]
        m3 = chromosome[2]
        m4 = chromosome[3]
        m5 = chromosome[4]
        m6 = chromosome[5]
        for i in range(6,12):
            if chromosome[i] == 0:
                chromosome[i] = chromosome[i] + 1
        d1 = chromosome[6]
        d2 = chromosome[7]
        d3 = chromosome[8]
        d4 = chromosome[9]
        d5 = chromosome[10]
        d6 = chromosome[11]
        x = []
        y = []
        # for n in range(1,self.plotResolution+1):
        #     x.append(n/self.plotResolution)
        #     y.append(n/self.plotResolution)
        for i in range(0,11):
            for j in range(0,11):
                x.append(i/11)
                y.append(j/11)

                mf1.append(pow(math.e,-0.5*pow((x[i]-m1)/d1,2)))
                mf2.append(pow(math.e,-0.5*pow((x[i]-m2)/d2,2)))
                mf3.append(pow(math.e,-0.5*pow((x[i]-m3)/d3,2)))

                mf4.append(pow(math.e,-0.5*pow((y[j]-m4)/d4,2)))
                mf5.append(pow(math.e,-0.5*pow((y[j]-m5)/d5,2)))
                mf6.append(pow(math.e,-0.5*pow((y[j]-m6)/d6,2)))
        
        fuzzy_X = np.arange(0, 11,11/121)
        self.ax3.plot(fuzzy_X, mf1, label = 'linear', color = "red")
        self.ax3.plot(fuzzy_X, mf2, label = 'linear', color = "blue")
        self.ax3.plot(fuzzy_X, mf3, label = 'linear', color = "green")
        
        self.ax3.plot(fuzzy_X, mf4, label = 'linear', color = "orange")
        self.ax3.plot(fuzzy_X, mf5, label = 'linear', color = "pink")
        self.ax3.plot(fuzzy_X, mf6, label = 'linear', color = "black")

        plt.show()
        plt.pause(0.01)

    
    def drawGraphs_Debugging(self, chromosome):
        plt.ion() # Turn on interactive mode using
        self.fig = plt.figure(figsize=plt.figaspect(0.25))
        self.fig.subplots_adjust(left=0.2, wspace=0.5)
        self.ax1 = self.fig.add_subplot(1, 3, 1, projection='3d') #Surfaces plot
        self.ax2 = self.fig.add_subplot(1, 3, 2)#Aptitude Function plot
        self.ax3 = self.fig.add_subplot(1, 3, 3)#Fuzzy Synthesis
        #self.fig.canvas.draw()
        #self.fig.canvas.flush_events()
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        # if self.mutation == True and self.elitism == False:
        #     plt.suptitle("Generación: " + str(generation+1) + ";" + " Población: " + str(population) + " individuos" + ";" + " Mutación: " + str(self.mutationPercentage)+ "% de la población" + ";" +" Elitismo:" + " No habilitado")
        # elif self.mutation == False and self.elitism == True:
        #     plt.suptitle("Generación: " + str(generation+1) + ";" + " Población: " + str(population) + " individuos" + ";" +" Mutación: " + "No habilitada " + ";" + " Elitismo:" + " Habilitado")
        # elif self.mutation == True and self.elitism == True:
        #     plt.suptitle("Generación: " + str(generation+1) + ";" + " Población: " + str(population) + " individuos" + ";" + " Mutación: " + str(self.mutationPercentage) + "% de la población" + ";" + " Elitismo:" + " Habilitado")
        # else:
        #     plt.suptitle("Generación: " + str(generation+1) + ";" + " Población: " + str(population) + " individuos" + ";" + " Mutación: " + "No habilitada " + ";" + " Elitismo:" + "No habilitado")

        self.ax1.set_xlabel('Fecha')
        self.ax1.set_ylabel('Colonia')
        self.ax1.set_zlabel('Precio del inmueble')
       # X = np.arange(1/self.plotResolution, 1 + 1/self.plotResolution, 1/self.plotResolution)
       # Y = np.arange(1/self.plotResolution, 1 + 1/self.plotResolution, 1/self.plotResolution)
        X = np.arange(0,11,1)
        Y = np.arange(0,11,1)
        X, Y = np.meshgrid(X, Y)
       # refZ = np.array(z1_RefValues) #Convert a Data Frame in an array
      #  z1_CalculatedValues = self.Create_TakagiSugenoSurface(chromosome)
      #  calculatedZ = np.array(z1_CalculatedValues)
      #  surf = self.ax1.plot_surface(X, Y, refZ, cmap= cm.coolwarm, linewidth=0)
      #  surf2 = self.ax1.plot_surface(X, Y, calculatedZ, cmap=cm.seismic, linewidth=0)

        # # Draw Aptitude Function
        # #plt.title(" Integral del error entre curvas ")
        # self.ax2.grid()
        # self.ax2.set_xlabel('Generación')
        # self.ax2.set_ylabel('Función de Aptitud')
        # self.ax2.plot(x2_Values, y2_Values, color='tab:orange',  marker='*')
        # self.ax2.text(x2_Values[len(x2_Values)-1], y2_Values[len(y2_Values)-1],"f(x)= " + str(round(y2_Values[len(y2_Values)-1],2)))

        #Draw Membership Functions
        self.ax3.title.set_text('Conjuntos Difusos')
        self.ax3.grid()
        self.ax3.set_xlabel('Conjuntos Difusos: X -> Fecha Y-> Colonia')
        self.ax3.set_ylabel('Grados de pertenencia')
        mf1 = []
        mf2 = []
        mf3 = []
        mf4 = []
        mf5 = []
        mf6 = []
        m1 = chromosome[0]
        m2 = chromosome[1]
        m3 = chromosome[2]
        m4 = chromosome[3]
        m5 = chromosome[4]
        m6 = chromosome[5]
        for i in range(6,12):
            if chromosome[i] == 0:
                chromosome[i] = chromosome[i] + 1
        d1 = chromosome[6]
        d2 = chromosome[7]
        d3 = chromosome[8]
        d4 = chromosome[9]
        d5 = chromosome[10]
        d6 = chromosome[11]
        x = []
        y = []
        # for n in range(1,self.plotResolution+1):
        #     x.append(n/self.plotResolution)
        #     y.append(n/self.plotResolution)
        for i in range(0,11):
            for j in range(0,11):
                x.append(i/11)
                y.append(j/11)

                mf1.append(pow(math.e,-0.5*pow((x[i]-m1)/d1,2)))
                mf2.append(pow(math.e,-0.5*pow((x[i]-m2)/d2,2)))
                mf3.append(pow(math.e,-0.5*pow((x[i]-m3)/d3,2)))

                mf4.append(pow(math.e,-0.5*pow((y[j]-m4)/d4,2)))
                mf5.append(pow(math.e,-0.5*pow((y[j]-m5)/d5,2)))
                mf6.append(pow(math.e,-0.5*pow((y[j]-m6)/d6,2)))
        
        fuzzy_X = np.arange(0, 11,11/121)
        self.ax3.plot(fuzzy_X, mf1, label = 'linear', color = "red")
        self.ax3.plot(fuzzy_X, mf2, label = 'linear', color = "blue")
        self.ax3.plot(fuzzy_X, mf3, label = 'linear', color = "green")
        
        self.ax3.plot(fuzzy_X, mf4, label = 'linear', color = "orange")
        self.ax3.plot(fuzzy_X, mf5, label = 'linear', color = "pink")
        self.ax3.plot(fuzzy_X, mf6, label = 'linear', color = "black")

        plt.show()
        plt.pause(0.01)
        input()


    def drawBestGraph(self,x1_Values,bestChromosome , generation, aptitudeFunction): #To do
        y1_Values = []
        for i in range(len(x1_Values)):
            y1_Values.append((bestChromosome[0]/self.peso)*((bestChromosome[1]/self.peso)*math.sin(x1_Values[i]/(bestChromosome[2]/self.peso))+(bestChromosome[3]/self.peso)*math.cos(x1_Values[i]/(bestChromosome[4]/self.peso)))+(bestChromosome[5]/self.peso)*x1_Values[i]-(bestChromosome[6]/self.peso)) 
        plt.clf()
        plt.suptitle("Mejor curva->Generación " + str(generation+1) +" f(x)-->Integral de error=" + str(round(aptitudeFunction,2)) + " Obtenidos: " + " A=" + str(round(bestChromosome[0]/self.peso)) + " B=" + str(round(bestChromosome[1]/self.peso)) + " C=" + str(round(bestChromosome[2]/self.peso)) + " D=" + str(round(bestChromosome[3]/self.peso)) + " E=" + str(round(bestChromosome[4]/self.peso)) + " F=" + str(round(bestChromosome[5]/self.peso)) + " G=" + str(round(bestChromosome[6]/self.peso)) + " Originales:" + " A=8" + " B=25" + " C=4" + " D=45" + " E=10" + " F=17" + " G=35")
        plt.xlabel("X", loc ="right")
        plt.ylabel("Y", loc= "top")
        plt.plot(x1_Values, self.Y_reference_List, color='tab:blue', label ="Curva de referencia")
        plt.plot(x1_Values, y1_Values, color= 'tab:orange', label = "Mejor curva por generación")
        plt.grid()
        plt.legend()
        plt.show()
        plt.pause(0.01)
    
    def getInitialPopulation(self, totalRows, totalColumns): 
        """Create the initial chromosome population"""
        population = []
        intList = list(range(256))
        shuffled = [0 for column in range(totalColumns)]
        for row in range(totalRows):
            shuffled = random.sample(intList, totalColumns)
            while shuffled[2] == 0 or shuffled[4] == 0:
                shuffled = random.sample(intList, totalColumns)
            population.append(shuffled) 
        return population

    def getAptitudeFunction(self, chromosome, referenceSurface):
        """Append the aptitude function at the end of chromosome. The Aptitude Function is calculated how absolute error"""
        TakagiSugenoSurface = self.Create_TakagiSugenoSurface(chromosome) #chromosome = [m1,m2,m3,m4,m5,m6,d1,d2,d3,d4,d5,d6,p1,p2,p3,p4,p5,p6,p7,p8,p9,q1,q2,q3,q4,q5,q6,q7,q8,q9,r1,r2,r3,r4,r5,r6,r7,r8,r9]
        errorArray = abs(referenceSurface - TakagiSugenoSurface)
        #print(f"errorArray= {errorArray}")
        columns= errorArray.columns
        aptitude_function = 0
        for col in columns:
            for row in range(len(errorArray[col])):
                aptitude_function = errorArray[col][row] + aptitude_function
        chromosome.append(aptitude_function)


    def selectChampion(self,totalOpponents, population, potentialOpponents ):
        "Return the champion in a tournament"
        totalColumns = len(population[0])
        totalRows = len(population)
        opponentsList = random.sample(potentialOpponents,totalOpponents) #Select opponents
        opponentsAptitudeFunctionsList = []
        for i in opponentsList:
            opponentsAptitudeFunctionsList.append(population[i][totalColumns-1])#Create list of Opponents Aptitude Function 
        aptitudeFunctionMin = min(opponentsAptitudeFunctionsList)
        champion = opponentsList[opponentsAptitudeFunctionsList.index(aptitudeFunctionMin)]
        return champion


    def performTournaments(self, population):
        """Return the tornaments champions per generation"""
        momsList = []
        dadsList = []
        totalColumns = len(population[0])
        totalRows = len(population)
        numOpponents = math.floor(totalRows * 0.05) #El grupo de contrincantes no debe de sobrepasar el 5% de la población.
        if totalRows%2 == 0:
            numRounds = int(totalRows/2)
        else:
            raise ValueError("The number of elements in the population is not even")
        
        championsList = [-1]       
        potentialOpponents = list(range(0,totalRows))

        for round in range(numRounds):

            if round == math.floor(numRounds/2):
                potentialOpponents = list(range(0,totalRows))
                championsList = [-1] 
            
            for opponent in potentialOpponents:
                for champion in championsList:
                    if opponent == champion:
                        index = potentialOpponents.index(opponent)
                        potentialOpponents.pop(index)
                        break

            momChampion = self.selectChampion(numOpponents, population, potentialOpponents)
            momsList.append(momChampion)
            championsList.append(momChampion)

            for opponent in potentialOpponents:
                for champion in championsList:
                    if opponent == champion:
                        index = potentialOpponents.index(opponent)
                        potentialOpponents.pop(index)
                        break

            dadChampion = self.selectChampion(numOpponents, population, potentialOpponents)
            dadsList.append(dadChampion)
            championsList.append(dadChampion)
        return momsList, dadsList


    def convertBinaryToDecimal(self, binaryNumber):#binaryNumber is a list
        decimal = 0
        for index in range(len(binaryNumber)):
            decimal = binaryNumber[index]*pow(2,len(binaryNumber)- 1 - index) + decimal
        return decimal


    def convertDecimalToBinary(self, decimalNumber):
        binary = []
        residue = 0
        if (decimalNumber - int(decimalNumber)) != 0:
            raise ValueError("El número debe ser entero") 
        elif decimalNumber <= -1:
            raise ValueError("El número debe ser mayor a 0")
        quotient = decimalNumber
        while(quotient != 0):
            residue =  quotient % 2
            binary.append(residue)
            quotient = int(quotient/2)
        if len(binary) < 8:
            zerosToFullBinary = 8 - len(binary)
            for i in range(zerosToFullBinary):
                binary.append(0)        
        binary.reverse()
        return binary


    def startReproductiveProcess(self, momsPopulation, dadsPopulation):
        totalColumns = len(momsPopulation[0])
        totalRows = len(momsPopulation)
        highMask = [0 for index in range(totalColumns*8)]
        lowMask = [0 for index in range(totalColumns*8)]
        momsAlleleList = [ [-1 for column in range(totalColumns*8)] for row in range(totalRows)]
        dadsAlleleList = [ [-1 for column in range(totalColumns*8)] for row in range(totalRows)]
        dadsLowSide = [ [-1 for index in range(totalColumns*8) ] for row in range(totalRows)]
        dadsHighSide = [ [-1 for index in range(totalColumns*8) ] for row in range(totalRows)]
        momsLowSide = [ [-1 for index in range(totalColumns*8) ] for row in range(totalRows)]
        momsHighSide = [ [-1 for index in range(totalColumns*8) ] for row in range(totalRows)]
        children = [ [-1 for index in range(totalColumns*8) ] for row in range(totalRows*2)]
        decimalChildren = [ [-1 for index in range(totalColumns) ] for row in range(totalRows*2)] 
        for chromosome in range(totalRows):
            for gen in range(totalColumns): # gen in decimal
                momBinaryNumber = self.convertDecimalToBinary(momsPopulation[chromosome][gen])
                dadBinaryNumber = self.convertDecimalToBinary(dadsPopulation[chromosome][gen])
                for x in range (len(momBinaryNumber)):
                    momsAlleleList[chromosome].pop(0)
                    momsAlleleList[chromosome].append(momBinaryNumber[x])
                    dadsAlleleList[chromosome].pop(0)
                    dadsAlleleList[chromosome].append(dadBinaryNumber[x])
        CuttingGene = random.randrange(1,(totalColumns*8-1))
        for index in range(len(highMask)):
            if index < CuttingGene:
                highMask[index] = 1
            else:
                lowMask[index] = 1
        #Cut chromosome to allele level
        for chromosome in range(totalRows):
            for allele in range(totalColumns*8):
                dadsLowSide[chromosome][allele] = dadsAlleleList[chromosome][allele] and lowMask[allele]
                dadsHighSide[chromosome][allele] = dadsAlleleList[chromosome][allele] and highMask[allele]
                momsLowSide[chromosome][allele] = momsAlleleList[chromosome][allele] and lowMask[allele]
                momsHighSide[chromosome][allele] = momsAlleleList[chromosome][allele] and highMask[allele]
        #Paste chromosome to allele level
        index = 0
        for chromosome in range(0,totalRows):
            for allele in range(totalColumns*8):
                children[index][allele] = dadsHighSide[chromosome][allele] or momsLowSide[chromosome][allele]   
                children[index+1][allele] = momsHighSide[chromosome][allele] or dadsLowSide[chromosome][allele]
            index +=2
        #Evaluate if mutation is enabled
        if self.mutation == True:
            children = self.mutatePopulation(children)
        #Convert alleles in genes
        for chromosome in range(len(children)):
            for gene in range(totalColumns):
                binaryGene = []
                for allele in range(8):
                    binaryGene.append(children[chromosome][gene*8 + allele])
                decimalGene = self.convertBinaryToDecimal(binaryGene)
                decimalChildren[chromosome][gene] = decimalGene 
        return decimalChildren


    def getBestAptitudeFunction(self, totalRows, totalColumns, population):
        bestResult = population[0][totalColumns]
        bestResult_Row = 0
        for row in range(1,totalRows): 
            if population[row][totalColumns] <=  bestResult: #Looking for the best aptitude Function. To Travelling Salesman Problem is the shorter route
                bestResult = population[row][totalColumns]
                bestResult_Row = row
        return bestResult, bestResult_Row

    def mutatePopulation(self, population):
        chromosomesPositions = list(range(len(population))) 
        percentage = round(len(population)*self.mutationPercentage*0.01)
        chromosomesToMutate= random.sample(chromosomesPositions, percentage)
        allelesPositions=list(range(len(population[0])))
        allelesToMutate = random.sample(allelesPositions,len(chromosomesToMutate))
        for chromosome, allele in zip(chromosomesToMutate, allelesToMutate):
                population[chromosome][allele] = int(not population[chromosome][allele])
        return population

    def applyElitism(self, parents, children):
        parentsAptitudeFunctions = []
        childrenAptitudeFunctions = []
        lastColumn = len(parents[0])-1
        parentsDictionary = {}
        childrenDictionary = {}
        bestChromosomes = [ [-1 for column in range(lastColumn+1)] for row in range(len(parents))]

        for chromosome in range(len(parents)):
            aptitudeFunction = parents[chromosome][lastColumn]
            parentsAptitudeFunctions.append(aptitudeFunction)
            parentsDictionary[aptitudeFunction] = chromosome

            aptitudeFunction = children[chromosome][lastColumn]
            childrenAptitudeFunctions.append(children[chromosome][lastColumn])
            childrenDictionary[aptitudeFunction] = chromosome

        parentsAptitudeFunctions.sort()
        childrenAptitudeFunctions.sort()
        half_Of_Population = round(len(parents)/2) #Half of the population
        for chromosome in range(half_Of_Population):
            parentChromosome = parentsDictionary[parentsAptitudeFunctions[chromosome]]
            bestChromosomes[chromosome] = parents[parentChromosome].copy()
        for chromosome in range(half_Of_Population):
            childrenChromosome = childrenDictionary[childrenAptitudeFunctions[chromosome]]
            bestChromosomes[chromosome + half_Of_Population] = children[childrenChromosome].copy()

        return bestChromosomes

    def getSubOptimalCurve(self,totalRows, totalColumns,totalGenerations):
        refSurface = self.createZ_ReferenceDataFrame()
        parents = self.getInitialPopulation(totalRows, totalColumns) 
        #Calculate Aptitude function by chromosome.
        #Each chromosome is built with 7 random numbers from 0 to 255. Each number is one of the seven coefficients of the equation
        for row in range(totalRows):
            self.getAptitudeFunction(parents[row], refSurface)  
        children =[ [-1 for column in range(totalColumns)] for row in range(totalRows)]
        momsChromosomes = [ [-1 for column in range(totalColumns)] for row in range(int(totalRows/2))]
        dadsChromosomes = [ [-1 for column in range(totalColumns)] for row in range(int(totalRows/2))]
        plt.ion() # Turn on interactive mode using
        self.fig = plt.figure(figsize=plt.figaspect(0.25))
        self.fig.subplots_adjust(left=0.2, wspace=0.5)
        self.ax1 = self.fig.add_subplot(1, 3, 1, projection='3d') #Surfaces plot
        self.ax2 = self.fig.add_subplot(1, 3, 2)#Aptitude Function plot
        self.ax3 = self.fig.add_subplot(1, 3, 3)#Fuzzy Synthesis

        betterAptitudeFunctionList = []
        generationList = []
        bestChromosome = []
        bestAptitudeFunction = 0
        bestGeneration = 0
        for generation in range(totalGenerations):
            xValues = []
            betterAptitudeFunction, betterResult_Row = self.getBestAptitudeFunction(totalRows, totalColumns,parents)
            betterChromosome = parents[betterResult_Row].copy()
            betterChromosome.pop() #Remove the aptitude function from the raw to leave only the genes
            betterAptitudeFunctionList.append(betterAptitudeFunction)
            generationList.append(generation)
            currentTakagiSugenoNet = self.Create_TakagiSugenoSurface(betterChromosome)
            #self.drawGraphs(refSurface,currentTakagiSugenoNet, generationList, betterAptitudeFunctionList, generation, totalRows)
            self.drawGraphs(refSurface,betterChromosome, generationList, betterAptitudeFunctionList, generation, totalRows)
            if bestAptitudeFunction == 0: #Only happens in Generation 1
                bestAptitudeFunction = betterAptitudeFunction
                bestChromosome = betterChromosome.copy()
                bestGeneration = generation
            elif bestAptitudeFunction > betterAptitudeFunction: #Looking for the suboptimal aptitude function
                bestAptitudeFunction = betterAptitudeFunction
                bestChromosome = betterChromosome.copy()
                bestGeneration = generation
            momsList,dadsList = self.performTournaments(parents) # Perform the tornaments
            for index in range(int(totalRows/2)):#Copy the  parents chromosomes 
                 momsChromosomes[index] = parents[momsList[index]].copy()
                 dadsChromosomes[index] = parents[dadsList[index]].copy()
                 if len(momsChromosomes[index]) > totalColumns:#Remove the aptitude function
                     momsChromosomes[index].pop(totalColumns)
                     dadsChromosomes[index].pop(totalColumns)
            children = self.startReproductiveProcess(momsChromosomes,dadsChromosomes ) # Start the reproductive process
            for row in range(totalRows):
                self.getAptitudeFunction(children[row], refSurface) #Calculate aptitude function for each child
            #Replace parent population by child population 
            if self.elitism == True:
                parents = self.applyElitism(parents, children)
            else:
                parents = children.copy()
       # plt.figure()
        #self.drawBestGraph(xValues,bestChromosome,bestGeneration, bestAptitudeFunction)
        input()


sa = SurfaceAdjustment(mutationPercentage = 0, elitism = False)
coeficientes = 39 #Parámetros de la red Takagi Sugeno
poblacion = 100
generaciones = 15
superficie = pd.read_csv("C:\\Users\\roxy_\\Documents\\MCC\\2022-1\\Inteligencia Artificial\\Proyecto\\DatosParaGraficarUnPlano.csv", index_col = "id")
print(f"Plano de referencia = {superficie}")
sa.getSubOptimalCurve(poblacion, coeficientes, generaciones)


#parametersList = [1,2,3,4,5,6,0,8,9,0,11,0,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30, 31,32,33,34,35,36,37,38,39]
#sa.Create_TakagiSugenoSurface(parametersList)

#chromosome = [0,0.5,1, 0,0.5,1, 0.21,0.21,0.21,0.21,0.21,0.21, 0,0,0,0,10,0,0,0,0,0,0,0,0,10,0,0,0,0,0,0,0,0,10,0,0,0,0]
#sa.drawGraphs_Debugging(chromosome)