# Real-Time Object Detection

# Para montar o video: cat video05-* > video05.mp4
# Para partir o arquivo (pedaços de 20MB) :  split -b 20000000 video05.mp4 video05-

# Imports
import cv2
import imutils
import time
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np

# Argumentos
prototxt='MobileNetSSD_deploy.prototxt.txt' # arquivo de deploy em Caffe
model='MobileNetSSD_deploy.caffemodel' # modelo pré-treinado
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

# Inicializa o video stream, inicia o sensor da câmera e inicializa o FPS counter
print("Iniciando o video stream...")

# Descomente esta linha para usar sua webcam. use src=0 para a webcam built-in e src=1 paar webcam USB
# vs = VideoStream(src=0).start()

# Usando arquivo de vídeo. Comente esta linha se for usar sua webcam
vs = VideoStream('video05.mp4').start()

# Inicia a captura e contabiliza os FPS (frames por segundo)
time.sleep(1.0)
fps = FPS().start() # frames por segundo

# Loop sobre os frames do video stream
while True:
	# Obtém o frame do video stream e redimensiona para ter uma largura máxima de 900 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=900)

	# Obtém as dimensões do frame e converte em um blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

	# Passa o blob através da rede e obtém as detecções e previsões
	net.setInput(blob)
	detections = net.forward()

	# Loop pelas detecções
	for i in np.arange(0, detections.shape[2]):
		# Extrai a confiança (isto é, a probabilidade) associada à previsão
		confidence = detections[0, 0, i, 2]

		# Filtra deteções fracas garantindo que a "confiança" seja maior do que a confiança mínima
		if confidence > confidence_min:
			# Extrai o índice do rótulo de classe das "detecções", então computa os coordenados (x, y) da caixa delimitadora para o objeto
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Desenha as previsões nos frames
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# Mostra o output
	cv2.imshow("Video Stream", frame)
	key = cv2.waitKey(1) & 0xFF

	# Se pressionada e tecla 'q' encerra o loop
	if key == ord("q"):
		break

	# Atualiza o FPS counter
	fps.update()

# Stop do timer
fps.stop()
print("Tempo total de captura: {:.2f}".format(fps.elapsed()))
print("FPS: {:.2f}".format(fps.fps()))

# Cleanup
cv2.destroyAllWindows()
vs.stop()