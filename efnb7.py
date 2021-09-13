# En este script se define la función que predice el resultado de una 
# red convlucional EffcientNet

# Importando bibliotecas necesarias
import efficientnet.keras as efn
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten
import os

# Temaño de la imagen de entrada
#size_img = 128


# Definiendo función get_img para leer los archivos numpy y ajustarlos a nuevo tamaño.
# La imagen que retorna la función también es normalizada sobre los valor máximo del pixel (255)
def get_img(imagen, size=128):
    
    img = np.expand_dims(imagen, axis=2)
    
    return tf.image.resize(img, [size,size], antialias=True)/255


# Función que calcula la salida del modelo EffcientNet
def efnb7_eval(X):

	dropout = 0.2

	# Creando el modelo EfficientNet
	rnet = efn.EfficientNetB7(weights=None, include_top=False, input_shape=[128,128,3])
	rnet = tf.keras.Sequential(rnet)
	rnet.add(Dropout(dropout))
	rnet.add(Flatten())
	rnet.add(Dense(4096, activation='relu'))
	rnet.add(Dropout(dropout))
	rnet.add(Dense(4096, activation='relu'))
	rnet.add(Dropout(dropout))
	rnet.add(Dense(1, activation='sigmoid'))

	# Cargando pesos finales
	rnet.load_weights(os.path.join('app','EFNB7_weights.h5'))

	# Estableiendo a un tama
	X_prueba = get_img(X)

	#X_prueba_128
	X_prueba_128 = np.zeros((1, 128, 128,3))

	# Almacenamos la imagen obtenida en cada canal
	X_prueba_128[0,:,:,0] = X_prueba[:,:,0]
	X_prueba_128[0,:,:,1] = X_prueba[:,:,0]
	X_prueba_128[0,:,:,2] = X_prueba[:,:,0]

	return rnet.predict(X_prueba_128)[0][0]
