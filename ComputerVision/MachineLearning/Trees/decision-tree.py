# Classificação de Imagens com Decision Tree e Random Forest

# Para juntar o dataset: cat 4scenes-* > 4scenes.zip
# Para separar o dataset: split -b 20000000 4scenes.zip 4scenes-

# Imports
import cv2
import mahotas # pre-processamento
import sklearn
import numpy as np
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Função para pré-processamento das imagens
def describe(image):

	# Extrair a média e o desvio padrão de cada canal da imagem no espaço de cores HSV
	(means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
	colorStats = np.concatenate([means, stds]).flatten()

	# Extrair recursos de textura Haralick
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	haralick = mahotas.features.haralick(gray).mean(axis=0) # utiliza textura com feature

	# Retorna um vetor de características concatenadas de estatísticas de cores e características de textura Haralick
	return np.hstack([colorStats, haralick])

# Argumentos
dataset='4scenes'
forest=0 # arvore de decisao=0, random forest=1

# Obtém o conjunto de caminhos de imagem e inicializa a lista de rótulos e matriz de recursos
print("\nExtraindo as Features...")
imagePaths = sorted(paths.list_images(dataset))
labels = []
data = []

# Loop sobre todas as imagens no diretório de entrada
for imagePath in imagePaths:
	# Extrai o rótulo e carrega a imagem a partir do disco
	label = imagePath[imagePath.rfind("/") + 1:].split("_")[0] # coast, forest, highway, street
	image = cv2.imread(imagePath)

	# Extrai recursos da imagem e atualiza a lista de rótulos e recursos
	features = describe(image) # extrai os descritores da imagem - X - dados de entrada
	labels.append(label)
	data.append(features)

# Construir o dataset de treino e de teste, com 75% dos dados para treinamento e 25% para testes
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data), np.array(labels), test_size=0.25, random_state=42)

# Inicializa o modelo como uma árvore de decisão
model = DecisionTreeClassifier(random_state=84)

# Verifica se Random Forest deve ser usado em vez disso
if forest > 0:
	model = RandomForestClassifier(n_estimators=20, random_state=42)

# Treinar a árvore de decisão
print("\nTreinando o Modelo..")
model.fit(trainData, trainLabels)

# Avaliando a classificação
print("\nAvaliando...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))

# Loop por imagens randômicas
for i in list(map(int, np.random.randint(0, high=len(imagePaths), size=(10,)))):
	imagePath = imagePaths[i]
	filename = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
	features = describe(image)
	prediction = model.predict(features.reshape(1, -1))[0]

	# Previsão
	print("\n[Previsão] {}: {}".format(filename, prediction))
	cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
