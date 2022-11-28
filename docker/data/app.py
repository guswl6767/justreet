import torch, base64
from PIL import Image
import json
from io import BytesIO
import pprint
import yolov5
import time
from urllib import request as req
import cv2
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications import vgg16

from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import img_to_array, array_to_img, load_img
# from keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
from flask import make_response,request
from flask import Flask,render_template,jsonify
from werkzeug.utils import secure_filename
import imghdr
app = Flask(__name__)

class Utill:
    #객체인식
    def ObjectDetecting(userurl):
        model = yolov5.load('./best.pt')

        userImg = req.urlopen(userurl).read()
        userImg = Image.open(BytesIO(userImg))

        uResults = model(userImg)
        uData = uResults.pandas().xyxy[0].to_json(orient="records")
        userData = json.loads(uData)

        return userData
    #풍경인식
    def SceneryDetection(userurl):
        classList = ['land','river_lake','mountain','ocean']
        resnetModel = tf.keras.models.load_model('./resNet.h5')

        userImg = req.urlopen(userurl).read()
        userImg = Image.open(BytesIO(userImg))
        userImg = userImg.resize((256,256))
        userImg = img_to_array(userImg)
        userImg = userImg.reshape((1, userImg.shape[0], userImg.shape[1], userImg.shape[2]))

        pred = resnetModel.predict(userImg)
        label = pred.argmax()
        name = classList[label]
        confidence = pred.max()

        return name, confidence

    def WaterName(name):
        if name in ['river_lake','ocean']:
            name = 'water'
        return name

    def ReturnMsg(status, msg, errType,data):
        returnMsg = {
            "Status":status,
            "Msg":msg,
            "Type": errType,
            "Data":data
        }
        return returnMsg


@app.route('/predict', methods=['POST'])
def handler():
    user = request.form.get('user',False)
    hostCategory = request.form.get('category',False)


    #풍경인식
    if hostCategory in ['1','2','3']:
        name, confidence = Utill.SceneryDetection(user)

    #객체인식
    else:
        userData = Utill.ObjectDetecting(user)
        if len(userData) == 0:
            data = {
                "Category":'etc'
            }
            return Utill.ReturnMsg(True, "CategoryDetection",1,data)
        name = userData[0]['name']
        confidence = userData[0]['confidence']

    #결과도출
    name = Utill.WaterName(name)
    data = {
            "Category":name,
            "Confidence":confidence * 100
    }
    response = Utill.ReturnMsg(True, "CategoryDetection",1,data)
    response = make_response(json.dumps(response, ensure_ascii=False))
    response.headers['Content-Type'] = 'multipart/form-data'

    return response


        
if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8888", debug=True)

