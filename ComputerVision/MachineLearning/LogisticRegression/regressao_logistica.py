# Classificação de Imagens com Regressão Logística
# Dataset: http://vis-www.cs.umass.edu/lfw/

#  Faces

# Imports
import cv2
import numpy as np
import imutils
import sklearn
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Começamos com um pequeno subconjunto das faces marcadas no conjunto de dados Wild e, em seguida, construímos as divisões de treinamento 
# e teste (nota: se esta for sua primeira vez executando esse script, pode demorar algum tempo para que o conjunto de dados seja baixado - 
# mas uma vez que ele baixou os dados serão armazenados em cache localmente e as execuções subsequentes serão substancialmente mais rápidas)
print("\nObtendo os dados...")
dataset = datasets.fetch_lfw_people(min_faces_per_person=70, funneled=True, resize=0.5) # dataset de celebridades, 70 faces por pessoa
(trainData, testData, trainLabels, testLabels) = train_test_split(dataset.data, dataset.target, test_size=0.25, random_state=42)

# Treina o modelo e mostra o Classification Report
print("\nTreinando o Modelo...")
model = LogisticRegression()
model.fit(trainData, trainLabels)
print(classification_report(testLabels, model.predict(testData), target_names=dataset.target_names))

# Loop sobre imagens randômicas
for i in list(map(int, np.random.randint(0, high=testLabels.shape[0], size=(10,)))):
    # Obtém uma imagem e o nome, depois redimensiona a imagem para que possamos visualizar melhor
    image = testData[i].reshape((62, 47))
    name = dataset.target_names[testLabels[i]]
    image = imutils.resize(image.astype("uint8"), width=image.shape[1] * 3, inter=cv2.INTER_CUBIC)

    # Classifica a Face
    prediction = model.predict(testData[i].reshape(1, -1))[0]
    prediction = dataset.target_names[prediction]
    print("Valor Previsto: {}, Valor Atual: {}".format(prediction, name))

    cv2.imshow("Previsto: {}, Atual: {}".format(prediction, name), image)
    #print(prediction)
    cv2.waitKey(0)