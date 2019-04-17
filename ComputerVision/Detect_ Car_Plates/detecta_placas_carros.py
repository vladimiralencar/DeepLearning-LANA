# Detecta Placas

# python mini-projeto5-02-detecta_placa.py --images testing_dataset --char-classifier output/simple_char.cpickle --digit-classifier output/simple_digit.cpickle

# Imports
import cv2
import imutils
import numpy as np
import argparse
import pickle
from imutils import paths
from tools.license_plate import LicensePlateDetector
from tools.descriptors import BlockBinaryPixelSum

# Argumentos
images='testing_dataset'
char_classifier='output/simple_char.cpickle'
digit_classifier='output/simple_digit.cpickle'

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", default=images, help="Caminho para o dataset com as imagens")
ap.add_argument("-c", "--char-classifier", default=char_classifier, help="Caminho para o classificador de caracteres")
ap.add_argument("-d", "--digit-classifier", default=digit_classifier, help="Caminho para o classificador de dígitos")
args = vars(ap.parse_args())

# Carrega os classificadores de caracteres e dígitos
charModel = pickle.loads(open(args["char_classifier"], "rb").read())
digitModel = pickle.loads(open(args["digit_classifier"], "rb").read())

# Inicializ o descritor
blockSizes = ((5, 5), (5, 10), (10, 5), (10, 10))
desc = BlockBinaryPixelSum(targetSize=(30, 15), blockSizes=blockSizes)

# Loop sobre as imagens
for imagePath in sorted(list(paths.list_images(args["images"]))):
	# Carrega a imagem
	print(imagePath[imagePath.rfind("/") + 1:])
	image = cv2.imread(imagePath)

	# Se a largura for maior que 640 pixels, então redimensiona a imagem
	if image.shape[1] > 640:
		image = imutils.resize(image, width=640)

	# Inicializa o detector da placa e detecta as placas e os caracteres
	lpd = LicensePlateDetector(image, numChars=7)
	plates = lpd.detect()

	# Loop sobre as placas detectadas
	for (lpBox, chars) in plates:
		lpBox = np.array(lpBox).reshape((-1, 1, 2)).astype(np.int32)

		# Inicializa o texto contendo os caracteres reconhecidos
		text = ""

		# Loop sobre cada caracter
		for (i, char) in enumerate(chars):
			# Pré-processa o caracter
			char = LicensePlateDetector.preprocessChar(char)
			if char is None:
				continue
			features = desc.describe(char).reshape(1, -1)

			# Se estes são os primeiros 3 caracteres, use o classificador de caracteres
			if i < 3:
				prediction = charModel.predict(features)[0]

			# Caso contrário, use o classificador de dígitos
			else:
				prediction = digitModel.predict(features)[0]

			# Atualiza o texto de caracteres reconhecidos
			text += prediction.upper()

		# Desenha os caracteres e a caixa delimitadora se houver alguns caracteres que possamos exibir
		if len(chars) > 0:
			# Calcula o centro da caixa delimitadora da placa
			M = cv2.moments(lpBox)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])

			# Desenha a região da placa e o texto da placa na imagem
			cv2.drawContours(image, [lpBox], -1, (0, 255, 0), 2)
			cv2.putText(image, text, (cX - (cX // 5), cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
				(0, 0, 255), 2)

	# Mostra o resultado
	cv2.imshow("Imagem", image)
	cv2.waitKey(0)