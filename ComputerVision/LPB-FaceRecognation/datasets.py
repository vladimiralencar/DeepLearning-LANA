# Módulo para Carregar o Dataset

# Imports
from sklearn.datasets.base import Bunch
from imutils import paths
from scipy import io
import numpy as np
import random
import cv2

def load_caltech_faces(datasetPath, min_faces=10, face_size=(47, 62), equal_samples=True,
	test_size=0.33, seed=42, flatten=False):
	# Obtém os caminhos da imagem associados às faces e carrega os dados da caixa delimitadora
	imagePaths = sorted(list(paths.list_images(datasetPath)))
	bbData = io.loadmat("{}/ImageData.mat".format(datasetPath))
	bbData = bbData["SubDir_Data"].T

	# Define a semente aleatória, inicializa a matriz de dados e os rótulos
	random.seed(seed)
	data = []
	labels = []

	# Loop pelas imagens
	for (i, imagePath) in enumerate(imagePaths):
		# Carrega a imagem e converte para Grayscale
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# Obtém a caixa delimitadora associada à imagem atual, extraia o ROI do rosto e redimensione-o para um tamanho canônico
		k = int(imagePath[imagePath.rfind("_") + 1:][:4]) - 1
		(xBL, yBL, xTL, yTL, xTR, yTR, xBR, yBR) = bbData[k].astype("int")
		face = gray[yTL:yBR, xTL:xBR]
		face = cv2.resize(face, face_size)

		# Verifica se o rosto deve ser "achatado" (flatten) em uma única linha
		if flatten:
			face = face.flatten()

		# Atualiza a matriz de dados e as etiquetas associadas
		data.append(face)
		labels.append(imagePath.split("/")[-2])

	# Converte a matriz de dados e a lista de etiquetas para uma matriz NumPy
	data = np.array(data)
	labels = np.array(labels)

	# Verifica se as amostras iguais para cada face devem ser usadas
	if equal_samples:
		# Inicializa a lista de índices 
		sampledIdxs = []

		# Loop pelos labels únicos
		for label in np.unique(labels):
			# Obtém os índices na matriz de etiquetas onde os rótulos são iguais ao rótulo atual
			labelIdxs = np.where(labels == label)[0]

			# Prossegue apenas se o número necessário de rostos mínimos puderem ser atendidos
			if len(labelIdxs) >= min_faces:
				# Aleatoriamente gera os índices para o rótulo atual, mantendo apenas o valor mínimo fornecido, e atualiza a lista de índices amostrados
				labelIdxs = random.sample(list(labelIdxs), min_faces)
				sampledIdxs.extend(labelIdxs)

		# Usa os índices amostrados para selecionar os pontos e os rótulos de dados apropriados
		random.shuffle(sampledIdxs)
		data = data[sampledIdxs]
		labels = labels[sampledIdxs]

	# Calcula o índice de divisão de treinamento e teste
	idxs = list(range(0, len(data)))
	random.shuffle(idxs)
	split = int(len(idxs) * (1.0 - test_size))

	# Divide os dados em segmentos de treinamento e teste
	(trainData, testData) = (data[:split], data[split:])
	(trainLabels, testLabels) = (labels[:split], labels[split:])

	# Cria os grupos de treinamento e teste
	training = Bunch(name="training", data=trainData, target=trainLabels)
	testing = Bunch(name="testing", data=testData, target=testLabels)

	# Devolve uma tupla do treinamento
	return (training, testing, labels)

	