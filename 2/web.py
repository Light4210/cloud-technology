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

def getClass(number):
    if (number == 1):
        return "Beach"
    if (number == 2):
        return "River"
    return "Something Else"

def convertImageToNP(image):
    image = image.read()
    image = Image.open(io.BytesIO(image))
    image = np.asarray(image)
    image = resize(image, (224,224,3))

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
    model_type = request.form['model']
    image = convertImageToNP(file)

    model = 0
    if (model == 'conv'):
        model = load_model('conv_net_model.h5')
    else:
        model = load_model('dense_net_model.h5')
    result = model.predict(image)

    result = np.argmax(result[0])
    if (result == 1):
        result = "Beach"
    if (result == 2):
        result = "River"
    else:
        result = "Something Else"
    return render_template('home.html', display="block", output=result, host=host)

app.run(host="0.0.0.0")