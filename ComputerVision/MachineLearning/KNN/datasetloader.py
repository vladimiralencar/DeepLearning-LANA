# Módulo para carregar os dados para o treinamento

# Imports
import cv2
import os
import numpy as np

# Classe DatasetLoader
class DatasetLoader:
	def __init__(self, preprocessors=None):
		self.preprocessors = preprocessors

		# Se os pré-processadores forem None, inicialize-os como uma lista vazia
		if self.preprocessors is None:
			self.preprocessors = []

	def load(self, imagePaths, verbose=-1):
		# Inicializa as listas de features e labels
		data = []
		labels = []

		# Loop pelas imagens de entrada
		for (i, imagePath) in enumerate(imagePaths):
			# Carrega a imagem e extrai o rótulo da classe supondo que nosso caminho tenha o seguinte formato:
			# /path/to/dataset/{class}/{image}.jpg
			image = cv2.imread(imagePath)
			label = imagePath.split(os.path.sep)[-2]

			# Verifique se nossos pré-processadores não são None
			if self.preprocessors is not None:
				# Superar os pré-processadores e aplicar cada um à imagem
				for p in self.preprocessors:
					image = p.preprocess(image)

			# Trate a nossa imagem processada como um "vetor de características" atualizando a lista de dados seguida pelos rótulos
			data.append(image)
			labels.append(label)

			# Mostra as atualizações
			if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
				print("Total de imagens processadas {}/{}".format(i + 1,
					len(imagePaths)))

		# Devolve uma tupla dos dados e rótulos
		return (np.array(data), np.array(labels))