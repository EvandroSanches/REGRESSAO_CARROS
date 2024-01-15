import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score
from pyod.models.knn import KNN
from pandasgui import show
import pickle

epochs = 40
batch_size = 100

def Carregar_Dados():
    #Preparando base de dados

    #Carregando dados
    dados = pd.read_csv('autos.csv', encoding='ISO-8859-1')

    #Removendo valores inconsistentes
    dados = dados[['price', 'vehicleType', 'yearOfRegistration', 'gearbox', 'powerPS', 'kilometer', 'fuelType', 'notRepairedDamage']]
    dados = dados.dropna()
    dados = dados[dados.price > 800]
    dados = dados[dados.price < 350000]

    #Codificando previsores com metodo OneHotEncoder (Previsores categoricos nominais)
    encoder = make_column_transformer(
                (OneHotEncoder(handle_unknown='ignore'), ['vehicleType','gearbox', 'fuelType', 'notRepairedDamage']),
                remainder='passthrough', sparse_threshold=False)
    dados = encoder.fit_transform(dados)

    #Removendo dados outliers
    dados = pd.DataFrame(dados)
    outliers = Outliers(dados)
    dados = dados.drop(index=outliers)

    #Separando dados previsores e target
    previsores = dados.drop(dados.columns[19], axis=1)
    target = dados.iloc[:, 19].values

    #Normalizando dados
    normalizador = MinMaxScaler()
    previsores = normalizador.fit_transform(previsores)

    return previsores, target


def Features(previsores, target):
    #Retorna um array com o valor de importancia de cada feature
    modelo = RandomForestClassifier()
    modelo.fit(previsores, target)
    resultado = modelo.feature_importances_

    return resultado
def Outliers(dados):
    #Encontra dados outliers através do algoritmo KNN
    detector = KNN()
    detector.fit(dados)

    labels = detector.labels_

    outliers = []

    #Pegando indice dos valores classificados como outlier
    for i in range(len(labels)):
        if labels[i] == 1:
            outliers.append(i)

    return outliers

def Treinamento():
    previsores, target = Carregar_Dados()

    modelo = HistGradientBoostingRegressor(max_iter=epochs)

    resultados = cross_val_score(estimator=modelo,
                                 X=previsores, y=target, cv=5, scoring='neg_mean_absolute_error')

    modelo.fit(previsores, target)

    with open('Modelo.0.1.pkl', 'wb') as file:
        pickle.dump(modelo, file)

    #Apresentando dados de treino
    media = resultados.mean()
    desvio = resultados.std()

    plt.plot(resultados)
    plt.title('Histórico de Treinamento\n'+'Média:'+str(media)+'\nDesvio:'+str(desvio))
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.show()

def Predict():
    previsores, target = Carregar_Dados()

    with open('Modelo.0.1.pkl', 'rb') as f:
        modelo = pickle.load(f)

    resultado = modelo.predict(previsores)

    target = np.expand_dims(target, axis=1)
    resultado = np.expand_dims(resultado, axis=1)

    resultado = np.concatenate((resultado, target), axis=1)

    predict = pd.DataFrame(data=resultado, columns=['Previsão', 'Preço'])
    predict['Diferença'] = predict.Preço - predict.Previsão

    show(predict)

Predict()
