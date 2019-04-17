# Treinamento do Classificador

# Imports
import cv2
import pickle
import imutils
import argparse
from imutils import paths
from sklearn.svm import LinearSVC
from tools.descriptors import BlockBinaryPixelSum

# Argumentos
ap = argparse.ArgumentParser()

fonts='input/example_fonts'
char_classifier='output/simple_char.cpickle'
digit_classifier='output/simple_digit.cpickle'
ap.add_argument("-f", "--fonts", default=fonts, help="Caminho para o dataset de fontes")
ap.add_argument("-c", "--char-classifier", default=char_classifier, help="Caminho para armazenar o classificador de caracteres")
ap.add_argument("-d", "--digit-classifier", default=digit_classifier, help="Caminho para armazenar o classificador de dígitos")
args = vars(ap.parse_args())

# Inicializa a string de caracteres
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"

# Inicializa os dados e os rótulos para o alfabeto e os dígitos
alphabetData = []
alphabetLabels = []
digitsData = []
digitsLabels = []

# Inicializa o descritor
print("Descrevendo as fontes de exemplo...")
blockSizes = ((5, 5), (5, 10), (10, 5), (10, 10))
desc = BlockBinaryPixelSum(targetSize=(30, 15), blockSizes=blockSizes)

# Loop sobre as fontes
for fontPath in paths.list_images(args["fonts"]):
	# Carrega a imagem da fonte, converte em escala de cinza e aplica threshold
	font = cv2.imread(fontPath)
	font = cv2.cvtColor(font, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(font, 128, 255, cv2.THRESH_BINARY_INV)[1]

	# Detecta contornos na imagem e classifica da esquerda para a direita
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	cnts = sorted(cnts, key=lambda c:(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[1]))

	# Loop sobre os contornos
	for (i, c) in enumerate(cnts):
		# Obtém a caixa delimitadora para o contorno, extrai o ROI e extrai recursos
		(x, y, w, h) = cv2.boundingRect(c)
		roi = thresh[y:y + h, x:x + w]
		features = desc.describe(roi)

		# Verifica se este é um caracter do alfabeto
		if i < 26:
			alphabetData.append(features)
			alphabetLabels.append(alphabet[i])

		# Caso contrário, é um dígito
		else:
			digitsData.append(features)
			digitsLabels.append(alphabet[i])

# Treina o classificador de caracteres
print("Treinando o classificador de caracteres...")
charModel = LinearSVC(C=1.0, random_state=42)
charModel.fit(alphabetData, alphabetLabels)

# Treina o classificador de dígitos
print("Treinando o classificador de dígitos...")
digitModel = LinearSVC(C=1.0, random_state=42)
digitModel.fit(digitsData, digitsLabels)

# Salvando o classificador de caracteres
print("Salvando o classificador de caracteres...")
f = open(args["char_classifier"], "wb")
f.write(pickle.dumps(charModel))
f.close()

# Salvando o classificador de dígitos
print("Salvando o classificador de dígitos...")
f = open(args["digit_classifier"], "wb")
f.write(pickle.dumps(digitModel))
f.close()

print("Treinamento concluído!")
print("\n")