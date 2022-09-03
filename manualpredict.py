import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

categories = ["not airplane", "airplane"]

def loadImage(filename):
	img = load_img(filename, target_size=(32, 32))
	img = img_to_array(img)
	img = img.reshape(1, 32, 32, 3).astype(int)
	plt.imshow(img[0])
	return img

def predictForImage(filename):
	img = loadImage(filename)
	# load model
	model = load_model('final_model.h5')
	result = model.predict(img)
	print("It is probably", categories[np.argmax(result, axis=-1)[0]], "with a", result[0][1], "chance of being", result[0][1])
	print(result)

