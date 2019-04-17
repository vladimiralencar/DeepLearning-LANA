# Detectando Um Único Objeto em Imagens com Deep Learning

# juntar os arquivos do modelo: cat bvlc_googlenet.caffemodel-* > bvlc_googlenet.caffemodel.zip
# descompactar: unzip bvlc_googlenet.caffemodel.zip
# separar o modelo em pedaços:  split -b 20000000 bvlc_googlenet.caffemodel.zip bvlc_googlenet.caffemodel-

# Imports
import cv2
import time
import numpy as np
from imutils import paths

# classifica a imagem com um modelo pré-treinado no Caffe
def dnn_classification(image):
    prototxt = 'bvlc_googlenet.prototxt'
    model = 'bvlc_googlenet.caffemodel'
    labels = 'synset_words.txt'

    # Carregando a imagem de entrada
    image = cv2.imread(image)

    # Carregando os labels
    rows = open(labels).read().strip().split("\n")
    classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

    # Nosso modelo pré-treinado é uma CNN e precisamos redimensionar a imagem de entrada para 224x224 pixels
    # e na sequência normalizar a imagem (para isso, vamos subtrair a média).
    # Ao final, o shape da imagem será: (1, 3, 224, 224) - Tamanho, Quant Canais, altura, largura da Imagem
    blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

    # Carregando o modelo
    print("\nCarregando o modelo pré-treinado...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # Passa a imagem de entrada como parâmetro, executar o forward e obtém a saída
    net.setInput(blob)
    start = time.time()
    preds = net.forward()
    end = time.time()
    print("Classificação feita em {:.5} segundos".format(end - start))

    # Ordena os índices das probabilidades em ordem descendente (probabilidade mais alta primeiro)
    # e imprime as 5 primeiras previsões
    idxs = np.argsort(preds[0])[::-1][:5]

    # Loop pelas previsões e print das imagens
    for (i, idx) in enumerate(idxs):
        # Imprime a previsão com maior probabilidade
        if i == 0:
            text = "Label: {}, {:.2f}%".format(classes[idx], preds[0][idx] * 100)
            cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Mostra o label previsto e a probabilidade associada
        print("{}. label: {}, probabilidade: {:.5}".format(i + 1,
                                                           classes[idx], preds[0][idx]))

    # Mostra o resultado final
    cv2.imshow("Imagem", image)
    cv2.waitKey(0)
    print("\n")

# le os arquivos de um folder
folder = 'demo'
for imagePath in paths.list_images(folder):
    dnn_classification(imagePath) # classifica as images
    # Exit se a tecla ESC for pressionada
    k = cv2.waitKey(1) & 0xff
    if k == 27: break

