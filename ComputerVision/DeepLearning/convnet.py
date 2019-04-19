# Imports
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
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
		mappings = {
			"vanilla": ConvNetFactory.VannillaNet,
			"lenet": ConvNetFactory.LeNet,
			"karpathynet": ConvNetFactory.KarpathyNet,
			"minivggnet": ConvNetFactory.MiniVGGNet}

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

	@staticmethod
	def LeNet(numChannels, imgRows, imgCols, numClasses, activation="tanh", **kwargs):
		model = Sequential()
		inputShape = (imgRows, imgCols, numChannels)

		# Se estivermos usando "channels_first", atualize a forma de entrada
		if K.image_data_format() == "channels_first":
			inputShape = (numChannels, imgRows, imgCols)

		# Define o primeiro set de CONV => ACTIVATION => POOL layers
		model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# Define o segundo set de CONV => ACTIVATION => POOL layers
		model.add(Conv2D(50, (5, 5), padding="same"))
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# Define o primeiro set de FC => ACTIVATION layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation(activation))

		# Define a segunda FC layer
		model.add(Dense(numClasses))

		# Finalmente a ativação softmax
		model.add(Activation("softmax"))

		# Retorna a arquitetura da rede
		return model

	@staticmethod
	def KarpathyNet(numChannels, imgRows, imgCols, numClasses, dropout=True, **kwargs):
		model = Sequential()
		inputShape = (imgRows, imgCols, numChannels)

		# Se estivermos usando "channels_first", atualize a forma de entrada
		if K.image_data_format() == "channels_first":
			inputShape = (numChannels, imgRows, imgCols)

		# Define o primeiro set de CONV => RELU => POOL layers
		model.add(Conv2D(16, (5, 5), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# Verifica se dropout deve ser aplicado para reduzir o overfitting
		if dropout:
			model.add(Dropout(0.25))

		# Define o segundo set de CONV => RELU => POOL layers
		model.add(Conv2D(32, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# Verifica se dropout deve ser aplicado para reduzir o overfitting
		if dropout:
			model.add(Dropout(0.25))

		# Define o terceiro set de  CONV => RELU => POOL layers
		model.add(Conv2D(64, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# Verifica se dropout deve ser aplicado para reduzir o overfitting
		if dropout:
			model.add(Dropout(0.5))

		# Define o primeiro set de FC => RELU layers
		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation("relu"))

		# Verifica se dropout deve ser aplicado para reduzir o overfitting
		if dropout:
			model.add(Dropout(0.5))

		# Define a segunda FC layer
		model.add(Dense(numClasses))

		# Finalmente a ativação softmax
		model.add(Activation("softmax"))

		# Retorna a arquitetura da rede
		return model

	@staticmethod
	def MiniVGGNet(numChannels, imgRows, imgCols, numClasses, dropout=True, **kwargs):
		model = Sequential()
		inputShape = (imgRows, imgCols, numChannels)
		chanDim = -1

		# Se estivermos usando "channels_first", atualize a forma de entrada
		if K.image_data_format() == "channels_first":
			inputShape = (numChannels, imgRows, imgCols)
			chanDim = 1

		# Define o primeiro set de  CONV => RELU => CONV => RELU => POOL layers
		model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# Verifica se dropout deve ser aplicado para reduzir o overfitting
		if dropout:
			model.add(Dropout(0.25))

		# Define o segundo set de CONV => RELU => CONV => RELU => POOL layers
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# Verifica se dropout deve ser aplicado para reduzir o overfitting
		if dropout:
			model.add(Dropout(0.25))

		# Define o primeiro set de FC => RELU layers
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))

		# Verifica se dropout deve ser aplicado para reduzir o overfitting
		if dropout:
			model.add(Dropout(0.5))

		# Define a segunda FC layer
		model.add(Dense(numClasses))

		# Finalmente a ativação softmax
		model.add(Activation("softmax"))

		# Retorna a arquitetura da rede
		return model

	