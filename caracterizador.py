# En este script definimos una clase que recibe una matriz (imagen)
# y retorna un

# Creado por Ricardo Niño de Rivera Barrón



#Importando bibliotecas necesarias
import numpy as np
from skimage.feature import greycomatrix
import scipy

class caracterizador:

	# Definimos el constructor de la clase
	# que recibe la imagen de interés (termograma) 
	# en escala de grises

	def __init__(self, imagen):
		self.imagen = imagen


	def prueba(self):

		# "Aplanando" la imagen
		flatten_image = self.imagen.flatten()

		# Eliminando los ceros del arreglo
		flatten_image = np.delete(flatten_image, np.where(flatten_image==0)[0])

		# Obteniendo "arreglos" que describen el histograma de la imagen
		image_hist=np.histogram(flatten_image, bins=255, range=(0, 256))

		# Con esta variable (que describe los valores en x del histograma) de calcularán algunos descriptores
		image_x=np.linspace(1,255,255)

		#Calculando el vector de probabilidad de la imagen
		p_image = image_hist[0]/flatten_image.shape[0]

		# Definimos un arreglo donde se almacenarán los valores de los descriptores
		features = np.zeros((1, 17))


		### DESCRIPTORES DEL HISTOGRAMA

		# Calculando media

		# Conforme a la ecuación propuesta en este trabajo mean1
		features[0,0]=np.sum(image_x*p_image)

		# Desviación Estándar

		# Con método de numpy std1
		features[0,1]=np.std(flatten_image, ddof=1)

		# Asimetría

		# Con método de scipy, asime
		features[0,2]=scipy.stats.skew(flatten_image)

		# Curtosis

		# Con método de scipy
		features[0,3]=scipy.stats.kurtosis(flatten_image, fisher=True)

		# Energía
		features[0,4] = np.sum(np.power(p_image, 2))

		# Entropía
		features[0,5] = scipy.stats.entropy(p_image, base=2)

		# Moda
		features[0,6] = scipy.stats.mode(flatten_image)[0][0]

		# Mediana
		features[0,7] = np.median(flatten_image)

		# Máximo
		features[0,8] = np.max(flatten_image)

		# Mínimo
		minimo=np.min(flatten_image)

		# Rango
		features[0,9] = features[0,8]-minimo

		### DESCRIPTORES DE LA IMAGEN DE CO-OCURRENCIA

		#Función para crear matriz de coocurrencia

		def p_coo(A):

			# Convertimos nuestra imagen a valores uint8
			A = A.astype(np.uint8)

			# Obtenemos la matriz de co-ocurrencia sin normalizar (histograma no probabilístico)
			coo = greycomatrix(A, [1], [0])[:,:,0,0]

			# Mandamos a cero el elemento 0,0
			coo[0,0] = 0

			# Matriz probabilística o normalizada
			p_coo = coo/np.sum(coo)

			return p_coo

		p_coo_image=p_coo(self.imagen)

		# Segundo Momento Angular o Energía
		features[0,10] = np.sum(np.power(p_coo_image,2))

		# Contraste
		for i in range(p_coo_image.shape[0]):
			for j in range(p_coo_image.shape[1]):
				features[0,11]+=np.power(i-j,2)*p_coo_image[i,j]

		# Correlación

		# Construyendo el vector de probabilidades p_x que recopila la probabilidad de aparición de los valores i
		# los valores i son la posición de referencia que se utilizan para la construcción de los pares ordenados i,j
		# en la matriz de co-ocurrencia.
		p_x=np.sum(p_coo_image, axis=1)

		# Construyendo el vector de probabilidades p_y que recopila la probabilidad de aparición de los valores j
		# los valores j a la dirección 0 grados que se utilizan para la construcción de los pares ordenados i,j
		# en la matriz de co-ocurrencia.
		p_y=np.sum(p_coo_image, axis=0)

		# Calculando la media de p_x
		mhu_x = np.sum(np.linspace(0,255,256)*p_x)

		# Calculando la media de p_y
		mhu_y = np.sum(np.linspace(0,255,256)*p_y)

		# Calculando desviación estándar de p_x
		sigma_x = np.sqrt(np.sum(p_x*np.power(np.linspace(0,255,256)-mhu_x, 2)))

		# Calculando desviación estándar de p_y
		sigma_y = np.sqrt(np.sum(p_y*np.power(np.linspace(0,255,256)-mhu_y, 2)))

		# Calculando el primer término de la covarianza
		covar1 = 0
		for i in range(p_coo_image.shape[0]):
			for j in range(p_coo_image.shape[1]):
				covar1 += i*j*p_coo_image[i,j]

		# Calculando la correlacion, correlation
		features[0,12] = (covar1 - mhu_x*mhu_y)/(sigma_x*sigma_y)

		# Varianza

		# Calculando mhu
		mhu = (mhu_x+mhu_y)/2

		# Calculando la varianza de p_coo
		for i in range(p_coo_image.shape[0]):
			for j in range(p_coo_image.shape[1]):
				features[0,13] += np.power(i-mhu,2)*p_coo_image[i,j]

		# Entropía

		# Calculando la entropía de p_coo_image
		for i in range(p_coo_image.shape[0]):
			for j in range(p_coo_image.shape[1]):
				# Para evitar errores agregamos está condición
				if p_coo_image[i,j] == 0:
					features[0,14] += 0
				else:
					features[0,14] += -p_coo_image[i,j]*np.log2(p_coo_image[i,j])

		# Varianza de la diferencia

		# Construyendo el vector px_y

		px_y = np.zeros((256,))

		k=0
		while k < 256:
			for i in range(p_coo_image.shape[0]):
				for j in range(p_coo_image.shape[1]):
					if k==np.abs(i-j):
						px_y[k]+=p_coo_image[i,j]
			k+=1

		# Calculando mhu_px_y
		mhu_px_y=np.sum(np.linspace(0,255,256)*px_y)

		# Calculando la varianza de la diferencia
		features[0,15] = np.sum(np.power(np.linspace(0,255,256)-mhu_px_y,2)*px_y)

		# Homogeneidad
		for i in range(p_coo_image.shape[0]):
			for j in range(p_coo_image.shape[1]):
				features[0,16] += p_coo_image[i,j]/(1+np.power(i-j,2))

		return features