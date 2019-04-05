# Detector de Faces
# Para executar estes script, use: cap04-01-face-detector

# Import
import cv2

# Carrega a imagem e converte para Grayscale

file = "royal.jpg"


import os

folder = 'demo2/'
image_names = sorted(os.listdir(folder))

i = 9
file = folder + image_names[i]


image = cv2.imread(file)
#cv2.imshow("Original", image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Carrega o detector de faces
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Detecta as faces
rects = detector.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 10, minSize = (30, 30))

# Loop por todas as faces e desenha um quadrado em torno delas
for (x, y, w, h) in rects:
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Mostra as faces detectadas
cv2.imshow("Faces", image)
cv2.waitKey(0)