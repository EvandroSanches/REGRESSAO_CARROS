import keras.optimizers
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
#from statsmodels.tools.tools import add_constant


def CriaRede():
    #Criando modelo
    modelo = Sequential()

    modelo.add(Dense(units=100, activation='leaky_relu', input_dim=315))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(units=100, activation='leaky_relu'))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(units=1, activation='linear'))

    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0018,
        decay_steps=70000,
        decay_rate=0.008
    )

    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_scheduler), loss='mae', metrics=['mae'])
    return modelo


def Carregar_Dados():
    #Preparando base de dados

    #Carregando dados
    dados = pd.read_csv('autos.csv', encoding='ISO-8859-1')

    #Removendo valores inconsistentes
    dados = dados.drop('dateCrawled',axis=1)
    dados = dados.drop('name',axis=1)
    dados = dados.drop('seller',axis=1)
    dados = dados.drop('offerType',axis=1)
    dados = dados.drop('dateCreated',axis=1)
    dados = dados.drop('nrOfPictures',axis=1)
    dados = dados.drop('postalCode',axis=1)
    dados = dados.drop('lastSeen',axis=1)
    dados = dados[dados.price > 250]
    dados = dados[dados.price < 350000]

    #Definindo alguns previsores vazios para o dado mais utilizado
    valores = {'vehicleType' : '', 'gearbox' : 'manuell',
            'model' : '', 'fuelType' : 'benzin',
            'notRepairedDamage' : 'nein'}

    dados = dados.fillna(value = valores)

    #Eliminando alguns registros com falta de dados
    dados = dados[dados.vehicleType != '']
    dados = dados[dados.model != '']

    #Separando dados previsores e target
    previsores = dados.drop('price', axis=1)
    target = dados.iloc[:, 0].values

    #Codificando previsores com metodo OneHotEncoder (Previsores categoricos nominais)
    encoder = make_column_transformer(
                (OneHotEncoder(handle_unknown='ignore'), ['abtest', 'vehicleType', 'gearbox', 'model', 'fuelType', 'brand','notRepairedDamage']),
                remainder='passthrough', sparse_threshold=False)
    previsores = encoder.fit_transform(previsores)

    #Salva encoding
    with open('modelo_onehotenc.pkl', 'wb') as file:
        pickle.dump(encoder, file)

    #df = pd.DataFrame(previsores, columns=encoder.get_feature_names_out())
    #A constant é adicionada através do bias no proprio modelo da rede neural
    #x = add_constant(df)

    return previsores, target



def GeraModelo():
    previsores, target = Carregar_Dados()
    modelo = CriaRede()

    modelo.fit(previsores, target, epochs=100, batch_size=300)
    modelo.save('Modelo.0.1')


def OneHotEncoding(data):
    #Carrega encoding
    try:
        encoder = pd.read_pickle('modelo_onehotenc.pkl')
        transformed = encoder.fit_transform(data)
        return transformed
    except:
        print('Treino o modelo para gerar o encoder')
        return




def Treinamento(previsores, target):

    modelo = KerasRegressor(build_fn=CriaRede,
                            epochs=100,
                            batch_size=300,
                            cv=10)


    resultados = cross_val_score(estimator=modelo,
                                 X=previsores, y=target, scoring='neg_mean_absolute_error')

    media = abs(resultados.mean())
    desvio = resultados.std()

    plt.bar(range(0,1), resultados)
    plt.title('Histórico de Treinamento\n'+'Média:'+str(media)+'\nDesvio:'+str(desvio))
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.show()


GeraModelo()

