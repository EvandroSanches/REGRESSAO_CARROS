import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer
from pandasgui import show

epochs = 40
batch_size = 100

def CriaRede():
    #Criando modelo
    modelo = Sequential()

    modelo.add(Dense(units=150, activation='relu', input_dim=22))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(units=150, activation='relu'))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(units=150, activation='relu'))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(units=1, activation='linear'))

    modelo.compile(optimizer='adam', loss='mae', metrics=['mse'])
    return modelo

def Carregar_Dados():
    #Preparando base de dados

    #Carregando dados
    dados = pd.read_csv('autos.csv', encoding='ISO-8859-1')

    #Removendo valores inconsistentes
    dados = dados[['price', 'vehicleType', 'yearOfRegistration', 'gearbox', 'powerPS', 'kilometer', 'fuelType', 'notRepairedDamage']]
    dados = dados.dropna()
    dados = dados[dados.price > 800]
    dados = dados[dados.price < 350000]

    #Separando dados previsores e target
    previsores = dados.drop('price', axis=1)
    target = dados.iloc[:, 0].values

    #Codificando previsores com metodo OneHotEncoder (Previsores categoricos nominais)
    encoder = make_column_transformer(
                (OneHotEncoder(handle_unknown='ignore'), ['vehicleType','gearbox', 'fuelType', 'notRepairedDamage']),
                remainder='passthrough', sparse_threshold=False)
    previsores = encoder.fit_transform(previsores)

    #Normalizando dados
    normalizador = MinMaxScaler()
    previsores = normalizador.fit_transform(previsores)

    return previsores, target


def Treinamento():
    previsores, target = Carregar_Dados()

    #Definindo callbacks
    es = EarlyStopping(monitor='loss', patience=10, verbose=1, min_delta= 1e-10)
    rlp = ReduceLROnPlateau(monitor='loss', patience=5, verbose=1)
    md_encoder = ModelCheckpoint(monitor='loss', filepath='Modelo.0.1', save_best_only=True, verbose=1)

    #Validação Cruzada
    modelo = KerasRegressor(build_fn=CriaRede,
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[es, rlp, md_encoder])


    resultados = cross_val_score(estimator=modelo,
                                 X=previsores, y=target, cv=5, scoring='neg_mean_absolute_error')

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

    modelo = load_model('Modelo.0.1')

    resultado = modelo.predict(previsores)

    target = np.expand_dims(target, axis=1)

    resultado = np.concatenate((resultado, target), axis=1)

    predict = pd.DataFrame(data=resultado, columns=['Previsão', 'Preço'])
    show(predict)

Predict()
