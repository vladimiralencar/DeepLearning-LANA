# SSD Para Detecção de Imagens

# Para juntar os arquivos: cat Archive-* > Archive2.zip
# Para separar em pedaços: split -b 20000000 Archive.zip Archive-


# Models: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

# https://github.com/lakshayg/tensorflow-build
# wget https://github.com/lakshayg/tensorflow-build/blob/master/tensorflow-1.4.0-cp36-cp36m-linux_x86_64.whl
# pip install tensorflow-1.4.0-cp36-cp36m-linux_x86_64.whl
# sudo apt-get install protobuf-compiler

# Se necessário habilitar o protoc, execute: protoc object_detection/protos/*.proto --python_out=.


# Imports
import os
import sys
import tarfile
import zipfile
import tensorflow as tf
import numpy as np
import six.moves.urllib as urllib
from PIL import Image
from io import StringIO
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from collections import defaultdict
from matplotlib import pyplot as plt
from object_detection.utils import ops as utils_ops


# Verifica se a versão do TensorFlow é 1.4
if tf.__version__ < '1.4.0':
  raise ImportError('Atualize sua instalação do tensorflow para v1.4. ou posterior!')

print("\nIniciando a detecção de objetos.....")

# Definindo o modelo para download
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'


# Caminho para o modelo
# Este é modelo que será usado nas previsões
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'


# Lista de labels que é usada para adicionar os labels para cada bounding box
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')


# Número de classes
NUM_CLASSES = 90

print("\nFazendo o download do modelo pré-treinado.....")

# Download do modelo escolhido
# Você pode comentar este bloco de código depois da primeira execução, pois não é necessário baixar o modelo cada vez que for fazer previsões
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


print("\nCarregando o modelo pré-treinado.....")


# Carregando o modelo
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# Carregando o map label
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Função para carregar uma imagem como array numpy
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# Caminho para as imagens
PATH_TO_TEST_IMAGES_DIR = 'imagens'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 7) ]

# Tamanho, em polegadas, das imagens de saída.
IMAGE_SIZE = (12, 8)


# Função para fazer inferência
def run_inference_for_single_image(image, graph): # previsão
  with graph.as_default():
    with tf.Session() as sess:
      
      # Obter handles para tensores de entrada e saída
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      
      for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      
      if 'detection_masks' in tensor_dict:
        
        # O processamento a seguir é apenas para imagem única
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        
        # Reframe é necessário para traduzir a máscara das coordenadas da caixa para as coordenadas da imagem e ajustar o tamanho da imagem
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        
        # Siga a convenção adicionando novamente a dimensão do lote
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
      
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Executando a previsão (inferência)
      output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

      # Todas as saídas são arrays float32 numpy, então converte os tipos conforme apropriado
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  
  return output_dict

print("\nDetecção de objetos em andamento.....")


for image_path in TEST_IMAGE_PATHS:
  image = Image.open(image_path)
  
  # A representação da imagem baseada em matriz será usada mais tarde para preparar a imagem do resultado com caixas e rótulos sobre ela.
  image_np = load_image_into_numpy_array(image)
    
  # Expandir as dimensões uma vez que o modelo espera que as imagens tenham forma: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
    
  # Detecção real
  output_dict = run_inference_for_single_image(image_np, detection_graph)
    
  # Visualização dos resultados de uma detecção
  vis_util.visualize_boxes_and_labels_on_image_array(
    image_np, 
    output_dict['detection_boxes'], 
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    instance_masks=output_dict.get('detection_masks'),
    use_normalized_coordinates=True,
    line_thickness=8)
    
  print("\nObjetos detectatos em", image_path)

  # Exibe a imagem na tela
  from scipy.misc import toimage
  toimage(image_np).show()

print("\nDetecção de objetos usando SSD concluída com sucesso!")
print("\n")


