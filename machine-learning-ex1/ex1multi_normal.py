"""
Regressao linear com multiplas variaveis 
(usando a equacao normal e sem normalizar)

Preço de casas a partir de duas variaveis:
x1: tamanho da casa
x2: numero de quartos
y: preco da casa   
"""

import pandas as pd
import numpy as np


# ------ Importando e arrumando os dados ------


#Importa os dados de um arquivo CSV
data = pd.read_csv(r'/home/bruno/Documents/Coursera/Machine Learning/machine-learning-ex1/ex1/ex1data2.txt', header=None)
#Transforma os dados em numpy array
data = data.to_numpy()

#Obtem o numero de linhas e colunas dos dados (m = linha, n = colunas)
m, n = data.shape

#Separando 
X = np.array(data[:,0:n-1])
y = np.array(data[:,2])
y = np.reshape(y,(m,1))

#Cria uma matriz para theta com valores iniciais de 0
theta = np.zeros((n, 1))

#Adiciona uma coluna de 1s na primeira coluna de X
ones = np.reshape(np.ones(m),(m,1))
X = np.concatenate((ones,X),axis=1)


# ------ Variaveis ----------------------------


#Numero de iteracoes e valor de alpha (learning rate)
iterations = 50
alpha = 0.3


# ------ Calculo da equacao normal ------------


##theta = inv(  X*(X.T)  ) *  X.T  *  y
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

for i in range(n):
    print("Theta", i,"=", round(float(theta[i]),4))


# ------ Prevendo novos valores ---------------


tamanho = 1650
quartos = 3

#Colocando os valores normalizados em uma matriz com a primeira coluna de 1s
matriz_teste = [1,tamanho,quartos] 

#Trocando as dimensoes da matriz
matriz_teste = np.reshape(matriz_teste,(1,n))

#Prevendo o valor fazendo matriz_teste x theta
valor_previsto = float(matriz_teste.dot(theta))

print("Valor previsto para tamanho de", tamanho, "sqr foot e", quartos, "quarto(s): $", round(valor_previsto,2))













