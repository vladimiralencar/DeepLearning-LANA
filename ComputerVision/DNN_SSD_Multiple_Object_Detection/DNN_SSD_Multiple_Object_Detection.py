# Detecção de Múltiplos Objetos em Imagens com Deep Learning

# python cap07-08-dnn_object_detection.py --image images/woman.jpg --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# Imports
import cv2
import numpy as np
from imutils import paths


prototxt='MobileNetSSD_deploy.prototxt.txt' # arquivo de deploy em Caffe
model='MobileNetSSD_deploy.caffemodel' # o modelo pré-treinado"
confidence_min = 0.2 # Probabilidade mínima para filtrar detecções fracas


# Inicializa a lista de labels MobileNet SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Carrega o modelo
print("\nCarregando o modelo...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

def dnn_SSD_classification(image):
	# Carrega a imagem de entrada e redimensiona para 300x300 e normaliza
	image = cv2.imread(image)
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

	# Passa a imagem pela rede para obter detecções e previsões
	print("Detectando objetos...")
	net.setInput(blob)
	detections = net.forward()

	# Loop pelas detecções
	for i in np.arange(0, detections.shape[2]):
		# Extrai a probabilidade associada com as previsões
		confidence = detections[0, 0, i, 2]

		# Filtra deteções fracas garantindo que a "confiança" seja maior do que a confiança mínima
		if confidence > confidence_min:
			# Extrai o índice do rótulo de classe das "detecções", então computa os coordenados (x, y) da caixa delimitadora para o objeto
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Mostra as previsões
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			print("Label {}".format(label))
			cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# Output
	cv2.imshow("Output", image)
	cv2.waitKey(0)
	print("\n")

# le os arquivos de um folder
folder = 'images'
for imagePath in paths.list_images(folder):
    dnn_SSD_classification(imagePath) # classifica as images
    # Exit se a tecla ESC for pressionada
    k = cv2.waitKey(1) & 0xff
    if k == 27: break
