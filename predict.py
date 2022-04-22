import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

longitud, altura = 150, 150
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  # print(array)
  contenido = os.listdir('./data/entrenamiento')
  print(contenido)
  print(contenido[answer])
  # print(answer)

predict("data/validacion/09 ARBOL/09 ARBOLCent1.jpg")