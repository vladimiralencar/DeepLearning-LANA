# Imports
import cv2

class Preprocessor:
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		# Armazena a largura da imagem alvo, a altura e o método de interpolação usado ao redimensionar
		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self, image):
		# Redimensione a imagem para um tamanho fixo, ignorando a relação de aspecto
		return cv2.resize(image, (self.width, self.height), interpolation=self.inter)