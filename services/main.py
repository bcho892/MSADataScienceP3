import io
import requests
import flask
import numpy as np
from PIL import Image
from categories import categories
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

def load_model():
    model = keras.models.load_model("final_model.h5")
    return model
    
app = flask.Flask(__name__)
model = load_model()

@app.route('/get_image')
def get_image_bytesio(): #https://stackoverflow.com/questions/68007907/return-an-image-taken-from-an-url-without-storing-the-image-file
    try:
        data = {"success" : False}
        r = requests.get(flask.request.args.get("imageUrl"))
        file = io.BytesIO()
        file.write(r.content)
        file.seek(0)
        image = Image.open(file)
        image = loadImage(image)
        information = predictForImage(model,image)
        data["classnum"] = information[0] 
        data["classname"] = information[1]
        data["probabilities"] = []
        for idx, prob in enumerate(information[2]):
            data["probabilities"].append({categories[idx]: str(prob)})
        data["success"] = True
        return flask.jsonify(data)
    except Exception as e:
        print(e)
        flask.abort(400)

def loadImage(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize((32,32))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3).astype(int)
    return img

def predictForImage(model, image):

    result = model.predict(image)
    mostLikelyClass = np.argmax(result, axis =-1)[0]
    className = categories[mostLikelyClass]
    return str(mostLikelyClass), className, result[0]
	
if __name__ == "__main__":
    print(("* Please wait we are loading"))
    load_model()
    app.run()