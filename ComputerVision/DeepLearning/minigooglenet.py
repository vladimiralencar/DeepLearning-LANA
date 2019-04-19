# Mini-GoogleLeNet

# Imports
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model  # --> Em outras arquiteturas usamos a função Sequential. Aqui usamos a função Model.
from keras.layers import concatenate
from keras import backend as K

class MiniGoogLeNet:
	@staticmethod
	def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):

		# x: A camada de entrada para a função.
		# K: O número de filtros que nossa camada CONV vai aprender.
		# kX e kY: o tamanho de cada um dos filtros K que serão aprendidos.
		# Stride: O passo da camada CONV.
		# chanDim: a dimensão do canal, que é derivada de "channels last" ou "channels first".
		# padding: O tipo de preenchimento a ser aplicado à camada CONV.

		x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Activation("relu")(x)

		return x

	@staticmethod
	def inception_module(x, numK1x1, numK3x3, chanDim):
		conv_1x1 = MiniGoogLeNet.conv_module(x, numK1x1, 1, 1, (1, 1), chanDim)
		conv_3x3 = MiniGoogLeNet.conv_module(x, numK3x3, 3, 3, (1, 1), chanDim)
		x = concatenate([conv_1x1, conv_3x3], axis=chanDim)

		return x

	@staticmethod
	def downsample_module(x, K, chanDim):
		conv_3x3 = MiniGoogLeNet.conv_module(x, K, 3, 3, (2, 2), chanDim, padding="valid")
		pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
		x = concatenate([conv_3x3, pool], axis=chanDim)

		return x

	@staticmethod
	def build(width, height, depth, classes):
		inputShape = (height, width, depth)
		chanDim = -1

		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		inputs = Input(shape=inputShape)
		x = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1), chanDim)

		# Dois módulos Inception seguidos por um módulo downsample 
		x = MiniGoogLeNet.inception_module(x, 32, 32, chanDim)
		x = MiniGoogLeNet.inception_module(x, 32, 48, chanDim)
		x = MiniGoogLeNet.downsample_module(x, 80, chanDim)

		# Quatro módulos Inception seguidos por um módulo downsample 
		x = MiniGoogLeNet.inception_module(x, 112, 48, chanDim)
		x = MiniGoogLeNet.inception_module(x, 96, 64, chanDim)
		x = MiniGoogLeNet.inception_module(x, 80, 80, chanDim)
		x = MiniGoogLeNet.inception_module(x, 48, 96, chanDim)
		x = MiniGoogLeNet.downsample_module(x, 96, chanDim)

		# Dois módulos Inception seguidos por AveragePooling2D e Dropout
		x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
		x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
		x = AveragePooling2D((7, 7))(x)
		x = Dropout(0.5)(x)

		# Classificador Softmax 
		x = Flatten()(x)
		x = Dense(classes)(x)
		x = Activation("softmax")(x)

		# Cria o modelo

		# Como estamos usando Model em vez de Sequential para definir a arquitetura de rede, não podemos chamar model.add, 
		# pois isso implicaria que a saída de uma camada segue sequencialmente para a próxima camada. 
		# Em vez disso, fornecemos a camada de entrada entre parênteses no final da chamada de função, que é chamada de API funcional. 
		# Cada instância da camada em um modelo é chamada em um tensor e também retorna um tensor. 
		# Portanto, podemos fornecer as entradas para uma determinada camada, chamando-a como uma função, uma vez que o objeto é instanciado.
		model = Model(inputs, x, name="googlenet")

		return model



		