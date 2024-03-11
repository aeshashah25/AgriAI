from flask import Flask, render_template, redirect, request, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import os
import tensorflow
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.preprocessing import OneHotEncoder


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'bucket')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}

MODEL = tensorflow.keras.models.load_model(os.path.join(MODEL_DIR, 'efficientnetv2s.h5'))
REC_MODEL = pickle.load(open(os.path.join(MODEL_DIR, 'RF.pkl'), 'rb'))

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CLASSES = ['Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Blueberry healthy', 'Cherry (including sour) Powdery mildew', 'Cherry (including sour) healthy', 'Corn (maize) Cercospora leaf spot Gray leaf spot', 'Corn(maize) Common rust', 'Corn(maize) Northern Leaf Blight', 'Corn(maize) healthy', 'Grape Black rot', 'Grape Esca(Black Measles)', 'Grape Leaf blight(Isariopsis Leaf Spot)', 'Grape healthy', 'Orange Haunglongbing(Citrus greening)', 'Peach Bacterial spot', 'Peach healthy', 'Bell PepperBacterial_spot', 'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Raspberry healthy', 'Soybean healthy', 'Squash Powdery mildew', 'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites (Two-spotted spider mite)', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']

@app.route('/')
def home():
        return render_template("index.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/about')
def aboutme():
    return render_template('about.html')

@app.route('/plantdisease/<res>')
def plantresult(res):
    print(res)
    corrected_result = ""
    for i in res:
        if i!='_':
            corrected_result = corrected_result+i
    return render_template('plantdiseaseresult.html', corrected_result=corrected_result)

@app.route('/plantdisease', methods=['GET', 'POST'])
def plantdisease():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            model = MODEL
            imagefile = tensorflow.keras.utils.load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(224, 224, 3))
            input_arr = tensorflow.keras.preprocessing.image.img_to_array(imagefile)
            input_arr = np.array([input_arr])
            result = model.predict(input_arr)
            probability_model = tensorflow.keras.Sequential([model, 
                                         tensorflow.keras.layers.Softmax()])
            predict = probability_model.predict(input_arr)
            p = np.argmax(predict[0])
            res = CLASSES[p]
            print(res)
            return redirect(url_for('plantresult', res=res))
    return render_template("plantdisease.html")

@app.route('/croprecommendation/<res>')
def cropresult(res):
    print(res)
    corrected_result = res
    return render_template('croprecresult.html', corrected_result=corrected_result)

@app.route('/croprecommendation', methods=['GET', 'POST'])
def cr():
    if request.method == 'POST':
        X = []
        if request.form.get('nitrogen'):
            X.append(float(request.form.get('nitrogen')))
        if request.form.get('phosphorous'):
            X.append(float(request.form.get('phosphorous')))
        if request.form.get('potassium'):
            X.append(float(request.form.get('potassium')))
        if request.form.get('temperature'):
            X.append(float(request.form.get('temperature')))
        if request.form.get('humidity'):
            X.append(float(request.form.get('humidity')))
        if request.form.get('ph'):
            X.append(float(request.form.get('ph')))
        if request.form.get('rainfall'):
            X.append(float(request.form.get('rainfall')))
        X = np.array(X)
        X = X.reshape(1, -1)
        res = REC_MODEL.predict(X)[0]
        # print(res)
        return redirect(url_for('cropresult', res=res))
    return render_template('croprec.html')



# #importing pickle files
# model = pickle.load(open('E:\\BE\\New folder\\Fertilizer-Prediction-main\\classifier.pkl','rb'))
# ferti = pickle.load(open('E:\\BE\\New folder\\Fertilizer-Prediction-main\\fertilizer.pkl','rb'))



# @app.route('/fpredict/<res>')
# def fresult(res):
#     print(res)
#     result = res
#     return render_template('f.html', result=result)

# @app.route('/fpredict', methods=['GET', 'POST'])
# def fpredict():
#     temp = request.form.get('temp')
#     humi = request.form.get('humid')
#     mois = request.form.get('mois')
#     soil = request.form.get('soil')
#     crop = request.form.get('crop')
#     nitro = request.form.get('nitro')
#     pota = request.form.get('pota')
#     phosp = request.form.get('phos')
#     input = [int(temp),int(humi),int(mois),int(soil),int(crop),int(nitro),int(pota),int(phosp)]

#     res = ferti.classes_[model.predict([input])]

#     return render_template('f.html',x = ('Predicted Fertilizer is {}'.format(res)))

# if __name__ == "__main__":
#     app.run(debug=True)



# Assuming 'model' and 'ferti' are loaded somewhere in your code
model = pickle.load(open('C:\\Users\\ASUS\\crop-recommendation-and-plant-disease-detection-main\\models\\recommender-models\\classifier.pkl','rb'))
ferti = pickle.load(open('C:\\Users\\ASUS\\crop-recommendation-and-plant-disease-detection-main\\models\\recommender-models\\fertilizer.pkl','rb'))

@app.route('/fpredict/<res>')
def fresult(res):
    print(res)
    result = res
    return render_template('f.html', result=result)

@app.route('/fpredict', methods=['GET', 'POST'])
def fpredict():
    temp = request.form.get('temp')
    humi = request.form.get('humid')
    mois = request.form.get('mois')
    soil = request.form.get('soil')
    crop = request.form.get('crop')
    nitro = request.form.get('nitro')
    pota = request.form.get('pota')
    phosp = request.form.get('phos')

    # Check if any of the values are None or empty strings
    if any(value is None or value == '' for value in [temp, humi, mois, soil, crop, nitro, pota, phosp]):
        return render_template('f.html', x='Please fill out all the fields.')

    try:
        # Convert to integers after making sure they are not None or empty
        input = [int(temp), int(humi), int(mois), int(soil), int(crop), int(nitro), int(pota), int(phosp)]
    except ValueError:
        return render_template('f.html', x='Invalid input. Please enter valid numbers.')

    # Make sure 'ferti.classes_' is a list or array
    classes_list = list(ferti.classes_)

    res = classes_list[model.predict([input])[0]]

    return render_template('f.html', x='Predicted Fertilizer is {}'.format(res))

if __name__ == "__main__":
    app.run(debug=True)



    