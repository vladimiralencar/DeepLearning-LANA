import streamlit as st
from PIL import Image
import math
import numpy as np
import pandas as pd
import tensorflow as tf

def prever_doencas_de_pele(model, image):

	folder = 'static/tmp/'
	url = folder + file

	dict_idx_doenca = {0: ['Actinic keratoses', 'Queratose Actínica'],
	 1: ['Basal cell carcinoma', 'Carcinoma de Células Basais' ],
	 2: ['Benign keratosis-like lesions ', 'Queratoses Benignas'],
	 3: ['Dermatofibroma', 'Dermatofibroma'], # (Histiocitoma Fibroso Benigno)' ],
	 4: ['Melanocytic nevi', 'Nevo Melanócito (Sinal)'], # (Nevo Pigmentado, Sinal)
	 5: [ 'Melanoma', 'Melanoma'],
	 6: ['Vascular lesions', 'Lesões de Pele Vasculares'],
	 7: ['Acne', 'Acne'],
	 8: ['AlopeciaAreata', 'Alopecia Areata']}


	indices = []
	doencas_en = []
	doencas_pt = []
	for idx, doenca in (dict_idx_doenca.items()):
		indices.append(idx)
		doencas_en.append(doenca[0])
		doencas_pt.append(doenca[1])

	img = image #Image.open(x)

	media_scale_image = 158.4125188825441
	std_scale_image = 47.42283803971779

	x_pred = np.asarray(img.resize((299,299)))
	x_pred = x_pred.reshape(1, 299, 299, 3)
	x_pred = (x_pred - media_scale_image) / std_scale_image
	pred = np.argmax(model.predict(x_pred), axis=-1)
	probs = model.predict(x_pred)[0]

	df = pd.DataFrame()
	df['probs'] = probs * 100.0
	df['probs'] = df['probs'].apply(lambda x : math.floor(x))
	df['doenca_en'] = doencas_en
	df['doenca_pt'] = doencas_pt
	df_ordenado = df.sort_values(by=['probs'], ascending=False).reset_index()

	# filtra lesoes prob > 0
	df_ordenado = df_ordenado[ df_ordenado['probs'] > 0.0]

	if len(df_ordenado) > 3:
		df_ordenado = df_ordenado[:3]

	doencas = df_ordenado['doenca_pt']
	probs = df_ordenado['probs']

	return doencas, probs





st.set_option('deprecation.showfileUploaderEncoding', False)

# footer {visibility: hidden;}

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            #footer {visibility: hidden;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.title("Classificação Doencas de Pele")
st.write("")

print('carregando o modelo...')
#file = 'Model_2021_CNN_VGG19-V01.hdf5'
file = 'Model_2021_CNN_Xception19-V02.hdf5'
file = "model-MacBookPro-i7-16GB-RAM.hdf5"
file = "Model_2021_CNN_Xception-V09.hdf5"
model = tf.keras.models.load_model(file)
print('modelo carregado.')

file_up = st.file_uploader("Carregue uma image", type="jpg")

if file_up is not None:
    image = Image.open(file_up)

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    doencas, probs = prever_doencas_de_pele(model, image)

    st.markdown("<h3>Diagnóstico</h3>", unsafe_allow_html=True)
    for doenca, prob in zip(doencas, probs):
        st.markdown("<h4>" + doenca + " - " + str(prob) + "%</h4>", unsafe_allow_html=True)


aviso = "ESTE RESULTADO NÃO SUBSTITUI A AVALIAÇÃO DO MÉDICO. Este é um sistema para auxílio diagnóstico de doenças da pele usando redes neurais. Todo o processamento é feito no seu dispositivo e as imagens não são enviadas para o seu servidor. Ao continuar, você assume toda a responsabilidade com o uso."


st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

#st.write(aviso)


#<*font color=‘red’>THIS TEXT WILL BE RED</*font>, unsafe_allow_html=True)

st.markdown(f'<div style="color: #856404; background-color: #fff3cd; border-color: #ffeeba;">{aviso}</div>',
			 unsafe_allow_html=True)
st.write("")
st.write("")
st.write("")
st.write("")

footer=" \
<style> your css code put here</style> \
\
<div class='footer'> \
\
<p></p> \
\
\
<p>Sistema de Apoio ao Diagnóstico de Lesões de Pele versão 1.0.5.</p> \
<p>Desenvolvido por Prof. Dr. Vladimir Costa de Alencar e Equipe de Pesquisadores do LANA/UEPB. Campina Grande, Paraíba, Brasil, 2021.</p> \
\
<p><a style='display:block;text-align:left;' \
\
href='https://www.valencar.com' target='_blank'>valencar@gmail.com</a></p> \
\
</div>"

st.markdown(footer, unsafe_allow_html=True) 

