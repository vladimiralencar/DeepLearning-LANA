# Object Tracking em Vídeo

# Execute assim, para fazer o tracking de objetos a partir da sua webcam: python object_tracking.py
# Execute assim, para fazer o tracking de objetos a partir de arquivos de vídeo: python object_tracking.py --video ball_green1.mp4

# Imports
import cv2
import argparse
import imutils

# Argumentos
ap = argparse.ArgumentParser()
video='ball_green2.mp4'
ap.add_argument("-v", "--video", default=video, help="Caminho (opcional) para o arquivo de vídeo")
args = vars(ap.parse_args())

# Definindo o range de cores que serão usadas para detectar objetos
colorRanges = [
	((29, 86, 6), (64, 255, 255), "bola verde")]

# Se um caminho de vídeo não foi fornecido, usamos a webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# Senão, usamos o arquivo de vídeo fornecido como parâmetro
else:
	camera = cv2.VideoCapture(args["video"])

# Loop
while True:
	# Obtém o frame corrente
	(grabbed, frame) = camera.read()

	# Se estivermos vendo um vídeo e não obtemos um frame, chegamos ao final do vídeo
	if args.get("video") and not grabbed:
		break

	# Redimensiona o frame, desfoca e converte para o espaço de cores HSV
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# Loop pelo range de cores
	for (lower, upper, colorName) in colorRanges:
		# Construímos uma máscara para todas as cores na faixa atual de HSV, então executamos uma série de dilatações e erosões 
		# para remover quaisquer pequenas bolhas deixadas na máscara
		mask = cv2.inRange(hsv, lower, upper)
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)

		# Encontramos os contornos da máscara
		(_, cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Somente prosseguimos, se pelo menos 1 contorno foi encontrado
		if len(cnts) > 0:
			# Encontramos o maior contorno na máscara, e então usamos para calcular o círculo mínimo de fechamento e centro do centróide
			c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			(cX, cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			# Basta desenhar o círculo delimitador e texto, se o raio atende um tamanho mínimo
			if radius > 10:
				cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
				cv2.putText(frame, colorName, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

	# Mostramos o frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# Se a tecla q for pressionada, interrompemos o loop
	if key == ord("q"):
		break

# Libera a câmera e fecha todas as janelas
camera.release()
cv2.destroyAllWindows()
