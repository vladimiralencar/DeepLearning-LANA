# Classificação de Imagens com KNN

# Juntar o dataset: cat animais-* > animais.zip
# Particionar o dataset: split -b 20000000 animais.zip animais-

# Execute: python knn_animais.py --dataset animais

# Download do dataset
# https://www.kaggle.com/c/dogs-vs-cats/data
#

# Imports
import argparse
from imutils import paths
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessor import Preprocessor
from datasetloader import DatasetLoader

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Caminho para o dataset de entrada")
ap.add_argument("-k", "--neighbors", type=int, default=1, help="Número de vizinhos para classificação")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# de jobs para a distância k-NN (-1 usa todos os núcleos disponíveis)")
args = vars(ap.parse_args())

# Obtém a lista de imagens
print("\nCarregando imagens...")
imagePaths = list(paths.list_images(args["dataset"]))

# Inicializa o pré-processador de imagem, carrega o conjunto de dados do disco e remove a matriz de dados
sp = Preprocessor(32, 32)
sdl = DatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# Mostra algumas informações sobre o consumo de memória das imagens
print("\nMatriz de Atributos (Features): {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

# Encode dos rótulos como números inteiros
le = LabelEncoder()
labels = le.fit_transform(labels)

# Particionar os dados em treinamento e testar divisões usando 75% dos dados para treinamento e os restantes 25% para testes
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Treinar e avaliar um classificador k-NN nas intensidades de pixel em bruto
print("\nAvaliando o Classificador KNN...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))


