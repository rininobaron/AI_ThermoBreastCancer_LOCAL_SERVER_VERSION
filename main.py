from flask import Flask, render_template, request
from base64 import b64encode
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from app import segmentador2
from app import caracterizador
import requests
import json
import traceback
from time import sleep
from celery import Celery
from celery.result import AsyncResult
from app.neural_net import neuralnet_eval
from app.efnb7 import efnb7_eval


# Función que crea un objeto tipo celery para realizar la tarea en segundo plano 
def make_celery(flask_app):
	celery = Celery(flask_app.import_name, backend=flask_app.config['result_backend'],
		broker=flask_app.config['CELERY_BROKER_URL'])
	celery.conf.update(flask_app.config)
	TaskBase = celery.Task
	class ContextTask(TaskBase):
		abstract = True
		def __call__(self, *args, **kwargs):
			with flask_app.app_context():
				return TaskBase.__call__(self, *args, **kwargs)
	celery.Task = ContextTask
	return celery

# Instanciando flask
app = Flask(__name__)

# Configurando flask app con celery
app.config.update(
	# Heroku
    #CELERY_BROKER_URL=os.environ['REDIS_URL'],
    #CELERY_RESULT_BACKEND=os.environ['REDIS_URL']
    # LOCAL
    CELERY_BROKER_URL='redis://localhost:6379',
    result_backend='redis://localhost:6379'
)
celery = make_celery(app)

# Carpeta de subida
#app.config['UPLOAD_FOLDER'] = './app/Termogramas_temporales'

cm_hot = mpl.cm.get_cmap('hot')

# CALCULANDO PCA

# Importando X_train para ajustar PCA
X_train = np.load('./app/X_train.npy')

# Importando el método PCA
from sklearn.decomposition import PCA

# Instanciando PCa en un objeto que transforme los datos a 3 dimensiones
pca = PCA(n_components=3)

# Función para retornar vector de carcterísticas después de aplicar PCA
def PCA_transform(vector):
    
    #Expandiendo una dimension
    #vector = np.expand_dims(vector,axis=0)
    
    #Transformando el vector
    vector_pca = pca.fit(X_train).transform(vector)
    
    return vector_pca

# Función para obtener vector de características
def caract(derecha, izquierda):

	#Caracterizando
	object_der=caracterizador.caracterizador(derecha)
	object_izq=caracterizador.caracterizador(izquierda)

	# Obteniendo los vectores con los descriptores
	vector_der=object_der.prueba()
	vector_izq=object_izq.prueba()

	# Calculando diferencia y almacenando
	vector_dif=np.abs(vector_der-vector_izq)

	return vector_dif


# Función que retorna los objetos temporales de memoria
# con el objetivo de mostrar las imágenes de interés sin
# utilizar almacenar las imágenes en la memoria ROM
def get_image(image, pseudo=True):

	if pseudo==True:
		arr = cm_hot(image)
	else:
		arr=image

	arr = np.uint8(arr * 255)

	img = Image.fromarray(arr.astype("uint8"))

	# Creando espacio de memoria para el objeto tipo archivp
	file_object = io.BytesIO()

	# Escribiendo el objeto tipo archivo como 'PNG'
	img.save(file_object, 'PNG')

	file_object.seek(0)

	mime = "image/png"

	img_base64 = b64encode(file_object.getvalue()).decode('ascii')

	uri = "data:%s;base64,%s"%(mime, img_base64)

	#print(uri)

	return uri


@app.route('/')
def index():

	return render_template('index.html')



# Se recibe la imagen y se redirige a una página que muestra ejecución de un proceso
@app.route('/upload', methods=['POST'])
def load():

	if request.method == 'POST':

		try:
			
			#obtenemos el archivo del input "archivo"
			f = request.files['archivo']
			#filename = secure_filename(f.filename)
			#f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			#Leyendo archivo
			arreglo=np.load(io.BytesIO(f.read()))
			print(arreglo.shape)

			# Convertimos el arreglo a lista para que pueda ser enviado como tipo json
			arreglo = arreglo.tolist()

			# Relizando tarea en segundo plano
			res = background.delay(arreglo)

			print("task_id: "+str(res.id))
			print(res.state)
			
			# Se muestra el template y se envía id de la tarea 
			return render_template('procesando.html', task_id=res.id)
		
		except Exception as e:

			traceback.print_exc()

			# Desplegando algunos errores comunes
			if e.__class__.__name__=="FileNotFoundError":
				mensaje1 = "No se encontró ningún archivo"
				mensaje2 =  "Recuerda seleccionar un archivo extensión .npy"
			else:
				mensaje1 = "Error desconocido"
				mensaje2 =  "Recuerda seleccionar un archivo extensión .npy"
			return render_template('error.html', mensaje1=mensaje1, mensaje2=mensaje2)


# Función del proceso realizado en segundo plano
@celery.task(bind=True, ignore_result=False)
def background(self, arreglo):

	self.update_state(state="PROGRESS", meta={'progress': 0, 'message':'Leyendo archivo'})

	sleep(5)

	# Leyendo archivos
	#files=os.listdir(os.path.join(app.config['UPLOAD_FOLDER']))
	#print(files)

	# Leemos el archivo
	#arr1 = np.load(os.path.join(app.config['UPLOAD_FOLDER'], files[0]), allow_pickle=True)

	# Conviertiendo la lista recibida a arreglo tipo numpy
	arr1 = np.array(arreglo)

	self.update_state(state="PROGRESS", meta={'progress': 10, 'message':'Segmentando Termograma'})

	sleep(5)

	# Instanciando segmentador final
	objeto_temp = segmentador2.segmentador(arr1)

	self.update_state(state="PROGRESS", meta={'progress': 15, 'message':'Caracterizando Segmentos'})

	sleep(3)

	# Obteniendo imágenes: máscara por otsu, lado derecho y lado izquierdo
	temp_mask, right_breast, left_breast = objeto_temp.prueba()

	# Caracterizando

	vect_dif = caract(right_breast, left_breast)

	self.update_state(state="PROGRESS", meta={'progress': 40, 'message':'Realizando PCA'})

	sleep(5)

	###MODELO1###

	# Aplicando PCA
	score_0 = PCA_transform(vect_dif)

	self.update_state(state="PROGRESS", meta={'progress': 50, 'message':'Evaluando Red Neuronal Tricapa'})

	sleep(5)

	try:

		# Redondeando a 2 decimales la probabilidad expresada en porcentaje
		resultado_1 = np.around((neuralnet_eval(score_0))*100, 2)

	except:

		resultado_1 = "¡Problemas con el modelo!"

	self.update_state(state="PROGRESS", meta={'progress': 65, 'message':'Evaluando EfficientNet'})

	sleep(5)

	###TERMINA MODELO1###


	###MODELO2###

	try:

		# Reasignando
		score_1 = arr1

		# Redondeando a 2 decimales la probabilidad expresada en porcentaje
		resultado_2 = np.around((efnb7_eval(score_1))*100, 2)

	except:

		sleep(5)

		resultado_2 = "¡Problemas con el modelo!"

	#TERMINA MODELO2#


	# Conviertiendo la máscara a tipo float por motivos
	# de visualización
	temp_mask = temp_mask.astype(float)

	self.update_state(state="PROGRESS", meta={'progress': 85, 'message':'Preparando imágenes finales'})

	sleep(5)

	# Conviertiendo arreglos de "lados" a enteros
	# para correcta visualización
	right_breast = right_breast.astype(int)
	left_breast = left_breast.astype(int)

	self.update_state(state="PROGRESS", meta={'progress': 95, 'message':'Ya casi terminamos'})

	sleep(5)

	# Los arreglos numpy no pueden enviarse como respuesta de una tarea tipo celery
	# por tanto son tranformados en listas
	arr1 = arr1.tolist()
	temp_mask = temp_mask.tolist()
	right_breast = right_breast.tolist()
	left_breast = left_breast.tolist()

	self.update_state(state="PROGRESS", meta={'progress': 100, 'message':'¡Hemos terminado!'})

	sleep(5)

	# Los sleeps son únicamente para propósitos de Experiencia de Usuario

	return arr1, temp_mask, right_breast, left_breast, resultado_1, resultado_2


# Función para devolver respuesta del estatus
@app.route('/check_status', methods=['GET', 'POST'])
def check_status():

	task_id = request.get_data(as_text=True)

	async_result = AsyncResult(task_id, app=background)

	if not async_result.ready():
		print(async_result.ready())
		print(async_result.info)
		print(async_result.info['progress'])
		return json.dumps({'finish': async_result.ready(), 'progress': async_result.info['progress'], 'message': async_result.info['message']})
	else:
		print(async_result.ready())
		print(async_result.info)
		return json.dumps({'finish': async_result.ready(), 'progress': 100, 'message': '¡Hemos terminado!'})


# Ruta de posible error en el proceso
@app.route('/error', methods=['POST'])
def error():

	try:

		# Recuperando ID de la tarea en celery
		task_id = request.form['task_id']

		# Finalizando tarea
		AsyncResult(task_id).revoke()

		# Mensaje de error
		mensaje1 = "Error desconocido"
		mensaje2 =  "Recuerda seleccionar un archivo extensión .npy"

		return render_template('error.html', mensaje1=mensaje1, mensaje2=mensaje2)

	except:

		# Mensaje de error
		mensaje1 = "Error desconocido"
		mensaje2 =  "Recuerda seleccionar un archivo extensión .npy"

		return render_template('error.html', mensaje1=mensaje1, mensaje2=mensaje2)


# Se muestran resultados
@app.route('/base', methods=['POST'])
def base():

	if request.method == 'POST':

		try:

			# Recueprando ID de la tarea en celery
			task_id = request.form['task_id']
			print("task_id: "+str(task_id))

			respuesta = AsyncResult(task_id, app=background)

			#while not respuesta.ready():
			#	sleep(1)
			#	print(respuesta.state)

			arr1, temp_mask, right_breast, left_breast, resultado_1, resultado_2 = respuesta.get()

			print(respuesta.state)

			# Conviertiendo las lista de respuesta a arreglos tipo numpy

			arr1 = np.array(arr1)
			temp_mask = np.array(temp_mask)
			right_breast = np.array(right_breast)
			left_breast = np.array(left_breast)

			# Preparamos los arreglos anteriores para su despliegue como imágenes

			uri = get_image(arr1)
			uri_ostu = get_image(temp_mask)
			uri_derecha = get_image(right_breast)
			uri_izquierda = get_image(left_breast)
			image_original = get_image(arr1, pseudo=False)
			
			# Borrando los archivos remanentes que pudiesen existir
			#files=os.listdir(os.path.join(app.config['UPLOAD_FOLDER']))
			#if len(files)!=0:
			#	for i in files:
			#		os.remove(os.path.join(app.config['UPLOAD_FOLDER'], i))

			return render_template('base.html', image=uri, image_otsu=uri_ostu,
				image_derecha=uri_derecha, image_izquierda=uri_izquierda, image_original=image_original,
				resultado_1=resultado_1 , resultado_2=resultado_2)

		except Exception as e:

			try:

				AsyncResult(request.form['task_id']).revoke()

			except:

				pass


			traceback.print_exc()

			# Borrando los archivos remanentes que pudiesen existir
			#files=os.listdir(os.path.join(app.config['UPLOAD_FOLDER']))
			#if len(files)!=0:
			#	for i in files:
			#		os.remove(os.path.join(app.config['UPLOAD_FOLDER'], i))

			print(e)
			print(e.__class__.__name__)

			# Desplegando algunos errores comunes
			if e.__class__.__name__=="FileNotFoundError":
				mensaje1 = "No se encontró ningún archivo"
				mensaje2 =  "Recuerda seleccionar un archivo extensión .npy"
			elif e.__class__.__name__=="ValueError":
				mensaje1 = "El archivo seleccionado no es válido"
				mensaje2 =  "Recuerda seleccionar un archivo extensión .npy"
			elif e.__class__.__name__=="KeyError":
				mensaje1 = "Existió un problema con IBM Watson Cloud"
				mensaje2 =  "Estamos trabajando para resolverlo"
			else:
				mensaje1 = "Error desconocido"
				mensaje2 =  "Recuerda seleccionar un archivo extensión .npy"
			return render_template('error.html', mensaje1=mensaje1, mensaje2=mensaje2)
