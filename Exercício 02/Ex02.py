import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error 

def randomForest(banda):
    
    diasASimular = 20
    totalDeDiasReais = banda.shape[0]

    xTrain = np.arange(0, len(banda['time'][0:totalDeDiasReais]))
    yTrain = banda['magnitude'][0:totalDeDiasReais]
    
    regressor = DecisionTreeRegressor()
    regressor.fit(xTrain.reshape(-1, 1), yTrain)
    
    xPredict = np.arange(0, len(xTrain + diasASimular))
    
    regressor = RandomForestRegressor(n_estimators = 100)
    regressor.fit(xTrain.reshape(-1, 1), yTrain)
    
    #predição
    y = regressor.predict(xPredict.reshape(-1, 1))
    
    magnitude = banda['magnitude'].to_numpy()
    
    erro.append(mean_squared_error(magnitude[0:totalDeDiasReais + diasASimular], y))
    
    return y
    
def arvoreDecisao(banda):
    
    diasASimular = 20
    totalDeDiasReais = banda.shape[0]

    xTrain = np.arange(0, len(banda['time'][0:totalDeDiasReais]))
    yTrain = banda['magnitude'][0:totalDeDiasReais]
    
    regressor = DecisionTreeRegressor()
    regressor.fit(xTrain.reshape(-1, 1), yTrain)
    
    xPredict = np.arange(0, len(xTrain + diasASimular))
    
    #predição
    y = regressor.predict(xPredict.reshape(-1, 1))
    
    time = banda['time'].to_numpy()
    magnitude = banda['magnitude'].to_numpy()
    
    erro.append(mean_squared_error(magnitude[0:totalDeDiasReais + diasASimular], y)) 
    
    return y

def regressaoGrau2(banda):
    
    diasASimular = 20
    totalDeDiasReais = banda.shape[0]

    xTrain = np.arange(0, len(banda['time'][0:totalDeDiasReais]))
    yTrain = banda['magnitude'][0:totalDeDiasReais]
    
    polynomialFeatures = PolynomialFeatures(degree = 2)
    XPoly = polynomialFeatures.fit_transform(xTrain.reshape(-1, 1))
    
    polyLinearRegression = LinearRegression()
    polyLinearRegression.fit(XPoly, yTrain)
    
    xPredict = np.arange(0, len(xTrain + diasASimular))

    #predição
    XPoly = polynomialFeatures.fit_transform(xPredict.reshape(-1, 1))
    y = polyLinearRegression.predict(XPoly)
    
    time = banda['time'].to_numpy()
    magnitude = banda['magnitude'].to_numpy()
    
    erro.append(mean_squared_error(magnitude[0:totalDeDiasReais + diasASimular], y))
    
    return y
    
def regressaoGrau3(banda):
    
    diasASimular = 20
    totalDeDiasReais = banda.shape[0]

    xTrain = np.arange(0, len(banda['time'][0:totalDeDiasReais]))
    yTrain = banda['magnitude'][0:totalDeDiasReais]
    
    polynomialFeatures = PolynomialFeatures(degree = 3)
    XPoly = polynomialFeatures.fit_transform(xTrain.reshape(-1, 1))
    
    polyLinearRegression = LinearRegression()
    polyLinearRegression.fit(XPoly, yTrain)
    
    xPredict = np.arange(0, len(xTrain + diasASimular))

    #predição
    XPoly = polynomialFeatures.fit_transform(xPredict.reshape(-1, 1))
    y = polyLinearRegression.predict(XPoly)
    
    time = banda['time'].to_numpy()
    magnitude = banda['magnitude'].to_numpy()
    
    erro.append(mean_squared_error(magnitude[0:totalDeDiasReais + diasASimular], y)) 
    
    return y

def regressaoLinear(banda):
    
    diasASimular = 20
    totalDeDiasReais = banda.shape[0]

    xTrain = np.arange(0, len(banda['time'][0:totalDeDiasReais]))
    yTrain = banda['magnitude'][0:totalDeDiasReais]

    xPredict = np.arange(0, len(xTrain + diasASimular))

    #treino
    LRModel = LinearRegression()
    try:
        LRModel.fit(xTrain.reshape(-1, 1), yTrain)
    except:
            pass

    #Predict
    y = LRModel.predict(xPredict.reshape(-1, 1))
    
    magnitude = banda['magnitude'].to_numpy()
    
    erro.append(mean_squared_error(magnitude[0:totalDeDiasReais + diasASimular], y)) 
    
    return y
    
#Pega a medida da luz em função do tempo da banda que iremos usar para treinar
def pegarBanda(base):
    
    #pegando a banda que tem mais informações dias medidos
    bandaComMaisDias = base['band'].value_counts().head(1)
    nomeBanda = bandaComMaisDias.index.tolist()
    
    base_banda = base.loc[(base['band']) == nomeBanda[0]]
    
    return base_banda

def grafico(p1, p2, p3, p4, p5, magnitude, tempo):
    
    plt.subplot(3,3,1),plt.scatter(tempo, magnitude, color = '#5de578', label = 'G')    
    plt.subplot(3,3,1),plt.plot(tempo, p1, color="red", label = "titulo")
    plt.subplot(3,3,1),plt.gca().invert_yaxis()
    
    plt.subplot(3,3,2),plt.scatter(tempo, magnitude, color = '#5de578', label = 'G')    
    plt.subplot(3,3,2),plt.plot(tempo, p2, color="red", label = "titulo")
    plt.subplot(3,3,2),plt.gca().invert_yaxis()
    
    plt.subplot(3,3,3),plt.scatter(tempo, magnitude, color = '#5de578', label = 'G')    
    plt.subplot(3,3,3),plt.plot(tempo, p3, color="red", label = "titulo")
    plt.subplot(3,3,3),plt.gca().invert_yaxis()
    
    
    
    plt.title('Fotometria SN2011fe')
    plt.xlabel('Tempo')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()

def main():
    
    #importando a base de dados
    base = pd.read_csv("sn5.csv")
    
    #separando a banda (comprimento de onda) que iremos utilizar
    banda = pegarBanda(base)  
    
    regressaoLinearPredict = regressaoLinear(banda)
    regressaoGrau2Predict = regressaoGrau2(banda)
    regressaoGrau3Predict = regressaoGrau3(banda)
    arvoreDecisaoPredict = arvoreDecisao(banda)
    randomForestPredict = randomForest(banda)
    
    magnitude = banda['magnitude'].to_numpy()
    tempo = banda['time'].to_numpy()
    
    grafico(regressaoLinearPredict, regressaoGrau2Predict, regressaoGrau3Predict, arvoreDecisaoPredict, randomForestPredict, magnitude, tempo) 
            
    print("Regressão Linear: ", erro[0])
    print("Regressão de Grau 2: ", erro[1])
    print("Regressão de Grau 3: ", erro[2])
    print("Árvore de Decisão: ", erro[3])
    print("Floresta Aleatória: ", erro[4])

erro = []
main()
















'''
plt.scatter(base_r['time'], base_r['magnitude'], color = 'red', label = 'R')
plt.scatter(base_v['time'], base_v['magnitude'], color = '#70c9d5', label = 'V')
plt.scatter(base_b['time'], base_b['magnitude'], color = 'blue', label = 'B')
plt.scatter(base_i['time'], base_i['magnitude'], color = '#ff7256', label = 'I')
plt.scatter(base_g['time'], base_g['magnitude'], color = '#5de578', label = 'G')
plt.gca().invert_yaxis()
plt.title('Magnitude')
plt.xlabel('Tempo')
plt.ylabel('Luz')
plt.legend()
plt.show()



    
    base_i = base.loc[(base['band'] == 'I') | (base['band'] == 'i')]
    base_g = base.loc[(base['band'] == 'G' )| (base['band'] == 'g')]
    base_r = base.loc[(base['band'] == 'R') | (base['band'] == 'r')]
    base_v = base.loc[(base['band'] == 'V') | (base['band'] == 'v')]
    base_b = base.loc[(base['band'] == 'B') | (base['band'] == 'b')]
    base_u = base.loc[(base['band'] == 'U') | (base['band'] == 'u')]
    base_z = base.loc[base['band'] == 'z']
    base_w1 = base.loc[base['band'] == 'W1']
    base_w2 = base.loc[base['band'] == 'W2']
    base_m2 = base.loc[base['band'] == 'M2']
    base_uvw1 = base.loc[base['band'] == 'UVW1']
    base_uvw2 = base.loc[base['band'] == 'UVW2']
    base_uvm2 = base.loc[base['band'] == 'UVM2']


'''























