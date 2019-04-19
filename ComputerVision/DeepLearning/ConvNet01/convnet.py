# Imports
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Sequential
from keras import backend as K

# Classe
class ConvNetFactory:
	def __init__(self):
		pass

	@staticmethod
	def build(name, *args, **kargs):
		# Define a rede
		mappings = {"vanilla": ConvNetFactory.VannillaNet}

		# Obtém a função construtor do dicionário de mapeamentos
		builder = mappings.get(name, None)

		# Se o construtor for None, então não haverá uma função que possa ser usada para construir na rede, portanto, retorne None
		if builder is None:
			return None

		# Se não, construa a rede com a arquitetura
		return builder(*args, **kargs)

	@staticmethod
	def VannillaNet(numChannels, imgRows, imgCols, numClasses, **kwargs):
		# Inicializa o modelo
		model = Sequential()
		inputShape = (imgRows, imgCols, numChannels)

		# Se estivermos usando "channels_first", atualize a forma de entrada
		if K.image_data_format() == "channels_first":
			inputShape = (numChannels, imgRows, imgCols)

		# Define o primeiro (e único) CONV => RELU layer
		model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))

		# Adiciona uma camada FC seguida do classificador soft-max
		model.add(Flatten())
		model.add(Dense(numClasses))
		model.add(Activation("softmax"))

		# Retorna a arquitetura da rede
		return model

	