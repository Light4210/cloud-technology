import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from skimage.transform import resize
from PIL import Image
import numpy as np
import base64
import cv2
import re
import os
import io
import math

app = Flask("__name__")

q = ""
path = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(path, 'static\predict.png')

def convertImageToNP(file):
    image = file.read()
    image = Image.open(io.BytesIO(image))
    image = np.asarray(image)
    image = image.astype('float32') / 255
    image = resize(image, (28,28,1))

    images = []
    images.append(image)
    return np.asarray(images)

@app.route("/")
def loadPage():
    host = request.host
    return render_template('home.html', display="none", host=host)

@app.route("/", methods=['POST'])
def predict():
    host = request.host
    file = request.files['file']
    image = convertImageToNP(file)

    model = load_model('model.h5')
    result = model.predict(image)
    result = result[0]

    result = result * 255.0
    result = result.astype(np.uint8)
    result = Image.fromarray(result)
    result.save(image_path)
    return render_template('home.html', display="", host=host)

app.run(host="0.0.0.0")