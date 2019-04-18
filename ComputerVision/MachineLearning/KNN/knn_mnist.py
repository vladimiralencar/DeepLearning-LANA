# Classificação de Imagens com KNN


# Imports
import cv2
import numpy as np
import imutils
import sklearn
from sklearn import datasets
from skimage import exposure
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Carrega o MNIST
mnist = datasets.load_digits()

# Obtém os dados MNIST e constrói a divisão de treinamento e testes, usando 75% dos dados para treinamento e 25% para testes
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data), mnist.target,
																  test_size=0.25, random_state=42)

# Agora, vamos tomar 10% dos dados de treinamento e usá-lo para validação
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1, random_state=84)

# Mostra a divisão dos dados
print("\n")
print("Dados de Treino: {}".format(len(trainLabels)))
print("Dados de Validação: {}".format(len(valLabels)))
print("Dados de Teste: {}".format(len(testLabels)))

# Inicializa os valores de k para o nosso classificador k-Nearest Neighbor juntamente com a lista de precisões para cada valor de k
kVals = range(1, 30, 2)
accuracies = []

print("\nLoop sobre os valores de K")
# Loop sobre os vários valores de `k` para o classificador do vizinho mais próximo
for k in range(1, 30, 2):
	model = KNeighborsClassifier(n_neighbors=k)
	model.fit(trainData, trainLabels)

	# Avaliar o modelo e atualizar a lista de precisão
	score = model.score(valData, valLabels)
	print("k=%d, acurácia=%.2f%%" % (k, score * 100))
	accuracies.append(score)

# Encontre o valor de k que tenha a maior precisão
i = int(np.argmax(accuracies))
print("\nQual o melhor Valor de K?")
print("k = %d conseguiu acurácia de %.2f%% nos dados de validação" % (kVals[i], accuracies[i] * 100))

# Re-treinar nosso classificador usando o melhor valor de k e prever os rótulos dos dados de teste
model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(trainData, trainLabels)
predictions = model.predict(testData)

# Mostra um relatório de classificação final que demonstre a precisão do classificador para cada um dos dígitos
print("\nAvaliação nos Dados de Teste")
print(classification_report(testLabels, predictions))

# Loop por dígitos de forma randômica
for i in list(map(int, np.random.randint(0, high=len(testLabels), size=(5,)))):
	# Obtém a imagem e classifica
	image = testData[i]
	prediction = model.predict(image.reshape(1, -1))[0]

	# Converte a imagem no formato array de 64 bits para uma imagem de 8 x 8 compatível com OpenCV, 
	# ajusta a intensidade dos pixels e então redimensiona para 32 x 32 pixels para que possamos vê-la melhor
	image = image.reshape((8, 8)).astype("uint8")
	image = exposure.rescale_intensity(image, out_range=(0, 255))
	image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

	# Mostra a previsão
	print("Eu acredito que este dígito é : {}".format(prediction))
	cv2.imshow("Digito previsto: " + str(prediction), image)
	cv2.waitKey(0)