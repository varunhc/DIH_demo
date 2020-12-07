import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import warnings
warnings.filterwarnings("ignore")
#warnings.filterwarnings('ignore', category=DeprecationWarning)
#warnings.filterwarnings('ignore', category=FutureWarning)
#import logging
#import tensorflow as tf
#tf.get_logger().setLevel(logging.ERROR) # disable Tensorflow warnings for this tutorial


import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import inference
import uuid

#from gevent.pywsgi import WSGIServer

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "srlw494$*Y(Wr"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS	

@app.route('/')
def upload_form():
    return render_template('upload.html')

#@app.route('/demo')
#def demoview():
#   return render_template('demo2.html')

#arr = os.listdir(UPLOAD_FOLDER)    
#for i in arr:
#    temp_rem = os.path.join(UPLOAD_FOLDER, i)
#    os.remove(temp_rem)
        

@app.route('/', methods=['POST'])
def upload_image():
    
    if 'file' not in request.files:
        flash('No file part')        
        return redirect(request.url)
    file = request.files['file']    
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):  
        filename = secure_filename(file.filename)        
        filename = uuid.uuid4().hex[:9]+"_"+filename[-12:]
        #print(filename)        
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file.save(fpath)
        flash('Image successfully uploaded and displayed')
        
        flash("Segmenting Cells and Analyzing...")      
        inference.infer(imgname=filename, img_dir = UPLOAD_FOLDER)
        flash("Inference Complete...")
        
        return render_template('upload.html', filename=filename, counter=str('count_'+filename),
        finalres=str('result_'+filename) )
    else:
        flash('Allowed image types are -> png, jpg, jpeg')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display/<counter>')
def display_count(counter):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + counter), code=301)

@app.route('/display/<finalres>')
def display_result(finalres):
    #print('display_image filename: ' + filesname)
    return redirect(url_for('static', filename='uploads/' + finalres), code=301)
    
if __name__ == "__main__":
    app.run()