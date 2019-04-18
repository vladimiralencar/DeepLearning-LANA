# Detecção de Objetos em Vídeos Usando SSD

# Para juntar os arquivos: cat Archive-* > Archive2.zip
# Para separar em pedaços: split -b 20000000 Archive.zip Archive-

# https://github.com/lakshayg/tensorflow-build
# wget https://github.com/lakshayg/tensorflow-build/blob/master/tensorflow-1.4.0-cp36-cp36m-linux_x86_64.whl
# pip install tensorflow-1.4.0-cp36-cp36m-linux_x86_64.whl

# sudo apt-get install protobuf-compiler

# Se necessário habilitar o protoc, execute: protoc object_detection/protos/*.proto --python_out=.

# pip install moviepy

# Abrir o shel do python e digitar:
# import imageio
# imageio.plugins.ffmpeg.download()

# Imports
import os
import sys
import imageio
import numpy as np
import tensorflow as tf
import object_detection
from PIL import Image
from datetime import datetime
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# Verifica se a versão do TensorFlow é 1.4
if tf.__version__ < '1.4.0':
  raise ImportError('Atualize sua instalação do tensorflow para v1.4. ou posterior!')


print("\nIniciando a detecção de objetos.....")


# Caminho para o diretório da API object_detection do TensorFlow
# https://github.com/tensorflow/models
# ****************** Altere o caminho abaixo para o diretório correspondente no seu computador ******************
MODELS_PATH = '.' #ssd_mobilenet_v1_coco_2017_11_17
   # '/Users/dmpm/Dropbox/DSA/ComputerVision/Cap07/08-SSD/ObjectDetection'
sys.path.append(MODELS_PATH)


#/Users/valencar/Dropbox/BigDataAnalytics/DeepLearning/DSA/VisaoComputacional/cap07-Videos/08-SSD/ObjectDetection/ssd_mobilenet_v1_coco_2017_11_17

# Considerando que já foi feito o download do modelo (verificar o script ssd_imagens.py)
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_CKPT = '%s/%s/frozen_inference_graph.pb' % (MODELS_PATH, MODEL_NAME)
PATH_TO_LABELS = '%s/data/mscoco_label_map.pbtxt' % MODELS_PATH
NUM_CLASSES = 90


# Carregando o modelo
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


print("\nCarregando o modelo pré-treinado.....")


# Carregando o map label
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Função para converter imagem em array numpy 
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


print("\nDetecção de objetos em andamento.....")


# Detecção de Objetos
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    
    # Tensores de entrada e saída definidos para detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    
    # Cada caixa representa uma parte da imagem onde um objeto específico foi detectado
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    
    # Cada score representa o nível de confiança para cada um dos objetos. 
    # A pontuação é mostrada na imagem do resultado, juntamente com o rótulo da classe.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')


    # Carrega o arquivo de vídeo

    # Nome do arquivo (sem a extensão)ca
    input_video = 'videos/video'

    # Leitura do arquivo
    video_reader = imageio.get_reader('%s.mp4' % input_video)

    # Grava um novo vídeo com os objetos detectados
    video_writer = imageio.get_writer('%s_objetos_detectados05.mp4' % input_video, fps=10)

    # Processa cada frame
    t0 = datetime.now()
    n_frames = 0
    for frame in video_reader:
      image_np = frame
      n_frames += 1

      # Expandir as dimensões uma vez que o modelo espera que as imagens tenham forma: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)

      # Detecção
      (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                               feed_dict={image_tensor: image_np_expanded})

      # Visualização dos resultados de uma detecção
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      
      # Gravando o vídeo com os objetos detectados
      video_writer.append_data(image_np)

    fps = n_frames / (datetime.now() - t0).total_seconds()
    print("\nFrames processados: %s, Velocidade: %s fps" % (n_frames, fps))
    print("\nDetecção de objetos em vídeo com SSD concluída com sucesso!")

    # Encerra a gravação
    video_writer.close()

