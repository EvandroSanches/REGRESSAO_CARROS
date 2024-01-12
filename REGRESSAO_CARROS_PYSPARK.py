# Databricks notebook source
#IMPORTAÇÕES
from pyspark.sql import SparkSession, functions as func
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, Normalizer
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

#CARREGANDO BASE DE DADOS
carros = spark.read.csv('/FileStore/tables/autos.csv', sep=',', header=True, inferSchema=True)
carros = carros.dropna()

# COMMAND ----------

#FILTRANDO VALORES E FEATURES
previsores = carros[["vehicleType", "yearOfRegistration", "gearbox", "powerPS", "kilometer", "notRepairedDamage", "price"]]
previsores = previsores.filter(previsores.price > 800)
previsores = previsores.filter(previsores.price < 350000)

# COMMAND ----------

#CODIFICANDO E NORMALIZANDO DADOS
index = StringIndexer(inputCols=['vehicleType','gearbox', 'notRepairedDamage'], outputCols=['vehicleType_index','gearbox_index', 'notRepairedDamage_index'])

encode = OneHotEncoder(inputCols=['vehicleType_index','gearbox_index', 'notRepairedDamage_index'], outputCols=['vehicleType_onehot','gearbox_onehot', 'notRepairedDamage_onehot'])

assemble = VectorAssembler(inputCols=['vehicleType_onehot','gearbox_onehot','notRepairedDamage_onehot','yearOfRegistration', 'powerPS', 'kilometer'], outputCol='features')

normalizador = Normalizer(inputCol='features', outputCol='features_scaled', p=1.0)

pipeline = Pipeline(stages=[index, encode, assemble, normalizador])
previsores = pipeline.fit(previsores).transform(previsores)
previsores = previsores['features_scaled', 'price']

# COMMAND ----------

#CRIANDO MODELO E VISUALIZANDO PREVISÕES
treino, teste = previsores.randomSplit([0.8, 0.2])
lr = LinearRegression(featuresCol='features_scaled', labelCol='price', predictionCol='predict')
lr_model = lr.fit(treino)
resultado = lr_model.transform(teste)
display(resultado)

# COMMAND ----------

#EVALUATE
evaluator = RegressionEvaluator(labelCol='price', predictionCol='predict', metricName='mae')
evaluate = evaluator.evaluate(resultado)
print(evaluate)
