import cv2 
import dlib
#Imutils nos ayudara a procesar imagenes y face utils para la deteccion de modelos
from imutils import face_utils 

img = cv2.imread('Img/avion.png')

#--------Modelos---------#
"""
Estos modelos se obtuvieron de: https://github.com/chuanqi305/MobileNet-SSD/

En el que prototxt contiene la estructura de la red neuronal, suas capas
y conexiones
Y caffemodel contiene la arquitectura del modelo y sus pesos ya
entrenados

Mientras que prototxt crea una construccion de la red neuronal,
caffemodel proporciona los datos para que la red pueda ejecutar 
la deteccion de imagenes, esto basado en la arquitectura SSD con MobileNet

"""
proto_file = 'SSD_MobileNet_prototxt'
model_file = 'SSD_MobileNet.caffemodel'

net = cv2.dnn.readNetFromCaffe(proto_file,model_file)

#------Variables para el modelo ---------#
""" Se procesan las imagenes para configurarlas segun los requisitos del modelo.

Estas son las imagenes que puede reconocer esta IA, debido a los modelos usados, es
por eso que no se pueden modificar, su orden o eliminar alguno de estos, simplemente
cambiar el nombre de las clases, si, el modelo ssd mobilenet fue entrenado para el reconocimiento
de imagenes, pero para un conjunto en especifico de clases con una asignacion previa de id"""
classNames = {0: 'background',
			  1:'avion', 2: 'bicicleta', 
			3: 'pajaro', 4: 'barco',
			5: 'bottela', 6: 'camion', 7: 'carro',
			8: 'gato', 9: 'silla',
			10: 'vaca', 11: 'mesa', 
			12: 'perro', 13: 'caballo',
			14: 'moto', 15: 'persona', 
			16: 'planta',
			17: 'oveja', 18: 'sofa', 
			19: 'tren', 20: 'tele'}
"""
Las imagenes se procesan aqui para hacerlas compatibles con el modelo ya
entrenado, ya que debe cumplir con los criterios de entrada del modelo
"""
input_shape = (300, 300) #Forma requerida para que entre al modelo
mean = (127.5, 127.5, 127.5) # Se normalizan los pixeles
scale = 0.007843 # Finalmente se escala

#---------Carga del modelo--------#
net = cv2.dnn.readNetFromCaffe(proto_file, model_file)

#------Procesamiento de la Imagen----#
blob = cv2.dnn.blobFromImage(img,
							scalefactor=scale,
							size=input_shape,
							mean=mean,
							swapRB=True) 
#——entrada de configuración—–#
net.setInput(blob)
"""
Se realiza la inferencia usando el modelo

Se dibujan rectangulos y etiquetas con el reconocimiento de imagen
"""
#—–usando el modelo para hacer predicciones
results = net.forward()

for i in range(results.shape[2]):

	# confidencia
	confidence = round(results[0, 0, i, 2],2) 
	if confidence > 0.7: #probabilidad de un 70% de ser detectado
	
		# class id del objeto
		id = int(results[0, 0, i, 1]) 
		
		# 3-6 contiene la coordenada que delimita la imagen
		x1, y1, x2, y2 = results[0, 0, i, 3:7] 
		
		# print(x1,y1,x2,y2)
		# Escala estas coordenadas sobre la imagen
		ih, iw, ic = img.shape
		x1, x2 = int(x1*iw), int(x2*iw)
		y1, y2 = int(y1 * ih), int(y2 * ih)
		cv2.rectangle(img,
					(x1, y1),
					(x2, y2),
					(0, 200, 0), 2) #Se dibuja una caja de img y texto, 
		 							#para el nombre  y confianza de detecion
		# Predicciones
		cv2.putText(img, f'{classNames[id]}:{confidence*100}',
					(x1, y1+20), 
					cv2.FONT_HERSHEY_DUPLEX, 
					1, (255, 0, 0), 1)
	# print(results[0,0,i,:])


img = cv2.resize(img, (600, 680))
cv2.imshow('Image', img)
# cv2.imwrite('output1.jpg',img) 
cv2.waitKey()
# Se hace uso de OpenCv para mostrar los resultados
