# En este script se define la función que predice el resultado de una 
# red neuronal tricapa

# Importando bibliotecas necesarias
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
import os

dropout = 0.2

# Creando el modelo de la red neuronal de tres capasocultas con 10,000 unidades por capa
rnet = tf.keras.Sequential()
rnet.add(Dense(300, activation="relu", input_shape=(3,)))
rnet.add(Dropout(dropout))
rnet.add(Dense(300, activation="relu"))
rnet.add(Dropout(dropout))
rnet.add(Dense(300, activation="relu"))
rnet.add(Dropout(dropout))
rnet.add(Dense(1, activation='sigmoid'))


# Cergando pesos finales
rnet.load_weights(os.path.join('app','rnet_weights.h5'))


def neuralnet_eval(X):

	dropout = 0.2

	# Creando el modelo de la red neuronal de tres capasocultas con 10,000 unidades por capa
	rnet = tf.keras.Sequential()
	rnet.add(Dense(300, activation="relu", input_shape=(3,)))
	rnet.add(Dropout(dropout))
	rnet.add(Dense(300, activation="relu"))
	rnet.add(Dropout(dropout))
	rnet.add(Dense(300, activation="relu"))
	rnet.add(Dropout(dropout))
	rnet.add(Dense(1, activation='sigmoid'))

	# Cergando pesos finales
	rnet.load_weights(os.path.join('app','rnet_weights.h5'))
	
	return rnet.predict(X)[0][0]


