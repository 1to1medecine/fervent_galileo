#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory
from pathlib import PosixPath
import fastai
from fastai import *
from fastai.vision import *
import matplotlib.pyplot as plt
from pylab import *
from PIL import Image

# On command line type:
# $ FLASK_APP=upload_image_DermMel.py flask run


print(os.getcwd())
UPLOAD_FOLDER = '/media/e33as/3A81-FD60/Hackathon/melanoma/DermMel_API/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
learn = load_learner('/media/e33as/3A81-FD60/Hackathon/melanoma/DermMel_API', fname='model_DermMel.pkl')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        print(file.filename)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            pathnb = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(pathnb)
            
            img = open_image(pathnb)
            pred_class,pred_idx,outputs = learn.predict(img)
            print('class: %s' %pred_class)
            return render_template('result.html', category=pred_class)
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)
