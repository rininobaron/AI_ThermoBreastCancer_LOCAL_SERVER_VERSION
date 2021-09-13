# En este script definimos una clase que recibe un archivo tipo npy
# que es la matriz previamente modificada para ser de 8 bits

# Y entraga dos matrices de las mamas derecha e izquierda
# Finalmente guarda las matrices resultantes en un directorio predeterminado
# Con formato npy

# Creado por Ricardo Niño de Rivera Barrón

#Importando bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi

class segmentador:

	# Definimos el constructor de la clase
	# Usamos la palabra para ser consistenetes con el código del Anexo C
	# temp es el arreglo tipo numpay con la matriz que define al termograma 
	# en escala de grises

	def __init__(self, imagen, display_images = False):
		self.temp = imagen
		self.display_images = display_images
 	
	def prueba(self):

		umbral=threshold_otsu(self.temp)
		#print("Umbral de la imagen de prueba: "+str(umbral))

		objeto_mask=self.temp>=umbral

		objeto=objeto_mask*self.temp

		#Creando array de una sola dimensión
		objeto=objeto.flatten()
		objeto=np.delete(objeto, np.where(objeto == 0))

		fondo_mask=self.temp<umbral
		fondo=fondo_mask*self.temp

		#Creando array de una sola dimensión
		fondo=fondo.flatten()
		fondo=np.delete(fondo, np.where(fondo == 0))

		#Imprimiendo el termograma con el umbral obtenido
		#plt.hist(fondo, bins=255, range=(0,255))
		#plt.hist(objeto, bins=255, range=(0,255))
		#plt.xlabel("Valor del Pixel")
		#plt.axvline(x=umbral)

		# Ahora vamos a binarizar la imagen con el umbral obtenido (máscara).
		temp_mask=objeto_mask
		temp_umbral=self.temp*objeto_mask



		#Imprimiendo imagen original
		if self.display_images==True:
			plt.figure()
			plt.title("Imagen original")
			plt.imshow(self.temp, cmap='hot')
			plt.show()

		#Imprimiendo mascara por Otsu
		if self.display_images==True:
			plt.figure()
			plt.title("Mascara obtenida con método de Otsu")
			plt.imshow(temp_mask, cmap='binary')
			plt.show()

		#Imprimiendo Imagen sin fondo
		if self.display_images==True:
			plt.figure()
			plt.title("Imagen sin el fondo")
			plt.imshow(temp_umbral, cmap='hot')
			plt.show()


		# Ahora encontramos el punto de la cintura de la forma descrita a continuación

		#Ubicando la posición del primero y el último True de la última fila de temp_mask
		cintura=np.where(temp_mask[temp_mask.shape[0]-1,:]==True)

		inicio=cintura[0][0]
		fin=cintura[0][cintura[0].shape[0]-1]

		intervalo_mitad=np.round((fin-inicio)/2)

		#indice mitad
		index_mitad=int(inicio+intervalo_mitad)

		# Creando las imágenes para cada lado (derecho e izquierdo) utilizando el plano sagital establecido por index_mitad

		# Incializando máscaras por lado
		mask_right=np.zeros((self.temp.shape[0],self.temp.shape[1]))
		mask_left=np.zeros((self.temp.shape[0],self.temp.shape[1]))


		# Máscaras por lado finales
		mask_right[:,:index_mitad]=1
		mask_left[:,index_mitad:]=1

		#Imágenes finales
		right_breast=self.temp*temp_mask*mask_right
		left_breast=self.temp*temp_mask*mask_left

		# Imprimiendo imágenes finales
		if self.display_images==True:
			fig, (ax1, ax2) = plt.subplots(1, 2)
			#fig.suptitle('Imágenes finales')
			ax1.imshow(right_breast, cmap='hot')
			ax1.title.set_text('Segmentación Lado Derecho')
			ax2.imshow(left_breast, cmap='hot')
			ax2.title.set_text('Segmentación Lado izquierdo')
			plt.show()

		return temp_mask, right_breast, left_breast














