"""
Regressão linear com multiplas variaveis (e normalização)

Preço de casas a partir de duas variaveis:
x1: tamanho da casa
x2: numero de quartos
y: preco da casa    
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ------ Funcoes ------------------------------


def custo(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    custo = np.sum(np.square(predictions-y))/(2*m)
    return custo


def gradiente_descendente(X,y,theta,alpha=0.01,iterations=100):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,3))
    for it in range(iterations):    
        prediction = np.dot(X,theta)
        theta = theta - (1/m)*alpha*(X.T.dot((prediction - y)))
        theta_history[it,:] = theta.T
        cost_history[it] = custo(theta,X,y)  
    return theta, cost_history, theta_history


# ------ Importando e arrumando os dados ------


#Importa os dados de um arquivo CSV
data = pd.read_csv(r'/home/bruno/Documents/Coursera/Machine Learning/machine-learning-ex1/ex1/ex1data2.txt', header=None)
#Transforma os dados em numpy array
data = data.to_numpy()

#Obtem o numero de linhas e colunas dos dados (m = linha, n = colunas)
m, n = data.shape

#Separando X e y
X = np.array(np.zeros((m,n-1)))
y = np.array(data[:,2])
y = np.reshape(y,(m,1))

#Cria uma matriz para theta com valores iniciais de 0
theta = np.zeros((n, 1))


# ------ Variaveis ----------------------------


#Numero de iteracoes e valor de alpha (learning rate)
iterations = 100
alpha = 0.3


# ------ Normalizando as variaves -------------


#Cria um vetor para as medias e um para o desvido padrao
mean = np.array(np.zeros((n-1)))
std = np.array(np.zeros((n-1)))

#Calcula a media de desvio padrao de todas as diferentes variaveis
for i in range(n-1):
    mean[i] = np.mean(data[:,i])
    std[i] = np.std(data[:,i])

#Normalizando as variaveis
for i in range(n-1):
    for j in range(m):
        X[j,i] = (data[j,i] - mean[i])/std[i]
       

#Adiciona uma coluna de 1s na primeira coluna de X
ones = np.reshape(np.ones(m),(m,1))
X = np.concatenate((ones,X),axis=1)


# ------ Chamada das funcoes ------------------


#Chamada do gradiente descendente
theta, cost_history, theta_history = gradiente_descendente(X,y,theta,alpha,iterations)


# ------ Plotando os dados --------------------


#Plotando os dados com valores padroes e normalizados 
plt.title("Sem normalizacao")
plt.scatter(data[:,0],data[:,1])
plt.show()
plt.title("Com normalizacao")
plt.scatter(X[:,1],X[:,2])
plt.show()

#Plotando o historico de custo
plt.title("Historico de custo")
plt.plot(cost_history)
plt.show()

#Plotando o historico de theta
plt.title("Historico de theta")
plt.plot(theta_history)
plt.show()

for i in range(n):
    print("Theta", i,"=", round(float(theta[i]),4))


# ------ Prevendo novos valores ---------------


tamanho = 1650
quartos = 3

#Normalizando os valores
tamanho_norm = (tamanho - mean[0])/std[0]
quartos_norm = (quartos - mean[1])/std[1]

#Colocando os valores normalizados em uma matriz com a primeira coluna de 1s
matriz_teste = [1,tamanho_norm,quartos_norm] 

#Trocando as dimensoes da matriz
matriz_teste = np.reshape(matriz_teste,(1,len(matriz_teste)))

#Prevendo o valor fazendo matriz_teste x theta
valor_previsto = float(matriz_teste.dot(theta))
print("Valor previsto para tamanho de", tamanho, "sqr foot e", quartos, "quarto(s): $", round(valor_previsto,2))


