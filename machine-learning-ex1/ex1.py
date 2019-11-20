"""
Regressao linear com uma variavel

Lucro de um restaurando dependendo da populacao da cidade:    
x1 = populacao de uma cidade
y = lucro na cidade
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.linear_model import LinearRegression
#import seaborn as sns


# ------ Funcoes ------------------------------


def custo(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    custo = np.sum(np.square(predictions-y))/(2*m)
    return custo
   
   
def gradiente_descendente(X,y,theta,alpha=0.01,iterations=100):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,2))
    for it in range(iterations):    
        prediction = np.dot(X,theta)
        theta = theta - (1/m)*alpha*(X.T.dot((prediction - y)))
        theta_history[it,:] = theta.T
        cost_history[it] = custo(theta,X,y)  
    return theta, cost_history, theta_history


# ------ Importando e arrumando os dados ------


#Importa os dados de um arquivo CSV
data = pd.read_csv(r'/home/bruno/Documents/Coursera/Machine Learning/machine-learning-ex1/ex1/ex1data1.txt', header=None)
#Transforma os dados em numpy array
data = data.to_numpy()

#Separa os dados em X e y
X = np.array(data[:,0])
y = np.array(data[:,1])

#Transforma o y em uma matriz len(y)x1
y = np.reshape(y,(len(y),1))

#Adiciona uma coluna de 1s na primeira coluna de X
X = np.transpose(np.vstack((np.ones(97),X)))

#Cria uma matriz para theta com valores iniciais de 0
theta = np.zeros((2, 1))

#Numero de iteracoes e valor de alpha (learning rate)
iterations = 1500
alpha = 0.01
    

# -------- Chamada das funcoes ----------------


#Chama a funcao gradiente descendente para calcular os valores de theta
theta, cost_history, theta_history = gradiente_descendente(X,y,theta,alpha,iterations)


# ------ Plotagem -----------------------------


##Para plotar usando o Seaborn, ja faz a reta
#data.columns = ['Pop','Preco']
#ax = sns.regplot(x="Pop", y="Preco", data=data)

#Plot dos dados 
plt.scatter(data[:,0],data[:,1])

#Plot da reta com os valores resultantes de theta
x_fun = X[:,1]
y_fun = theta[0,0] + theta[1,0]*x_fun
plt.plot(x_fun,y_fun,c='r')
plt.show()

#Plot do historico dos valores de custo
plt.title("Historico de custo")
plt.plot(cost_history)
plt.show()

#Plotando o historico de theta
plt.title("Historico de theta")
plt.plot(theta_history)
plt.show()


# -------- Prevendo novos valores  ------------


#Valor da populacao (dividido por 10.000), para o qual o lucro vai ser previsto
valor_prever = 3.5

#Colocando o valor em uma matriz com 1 na primeira coluna
valor_prever2 = [1,valor_prever]
valor_prever2 = np.reshape(valor_prever2,(1,len(valor_prever2)))

predicao = float(valor_prever2.dot(theta))
print("Para uma populacao de", valor_prever*10000, ", eh previsto um lucro de $", round(predicao*10000,2))


# --------- Usando o SkLearn ------------------

#
##Importa os dados de um arquivo CSV
#data = pd.read_csv(r'/home/bruno/Documents/Coursera/Machine Learning/machine-learning-ex1/ex1/ex1data1.txt', header=None)
##Transforma os dados em numpy array
#data = data.to_numpy()
#
##Separa os dados em X e y
#X = np.array(data[:,0]).reshape(-1,1)
#y = np.array(data[:,1])
#
##Criando uma instancia da classe LinearRegression e acha os valores de theta com fit
#model = LinearRegression().fit(X, y)    
#
##Analisando o 'coefficient of determination' (R²) 
#r_sq = model.score(X, y)
#print('coefficient of determination:', r_sq)
#
##Valores de Theta 0 e Theta 1 (.intercept_ eh um escalar, .coef_ eh um array)
#print('Theta 0 (intercept):', model.intercept_)
#print('Theta 1 (slope):', model.coef_)
#
##Prevendo resultados
#y_pred = model.predict(X)
#print('predicted response:', y_pred, sep='\n')



