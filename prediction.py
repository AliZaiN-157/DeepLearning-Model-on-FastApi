from PIL import Image
import numpy as np
from tensorflow.keras import preprocessing
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model

# def load_model():
#     CNN = load_model('D:\work\FastApi_DL_Project\Fire_and_Smoke_model.h5')
#     return CNN
# classifier = load_model()


def predict(image):
    CNN = load_model(
        'D:\work\FastApi_DL_Project\Fire_and_Smoke_model.h5')
    shape = ((256, 256, 3))
    model = tf.keras.Sequential([hub.KerasLayer(CNN, input_shape=shape)])
    test_image = image.resize((256, 256))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image/255.0
    test_image = np.expand_dims(test_image, axis=0)
    predictions = model.predict(test_image)

    return predictions
