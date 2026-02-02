import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd


# Create uploads directory if it doesn't exist
os.makedirs('static/uploads', exist_ok=True)
print("Ensuring static/uploads directory exists")

# Load CSV files
try:
    disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
    supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')
    print("CSV files loaded successfully")
except Exception as e:
    print(f"Error loading CSV files: {e}")

# Load model
try:
    model = CNN.CNN(39)
    model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

def prediction(image_path):
    try:
        print(f"Processing image: {image_path}")
        image = Image.open(image_path)
        # Convert image to RGB format
        if image.mode != 'RGB':
            print(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        image = image.resize((224, 224))
        input_data = TF.to_tensor(image)
        input_data = input_data.view((-1, 3, 224, 224))
        output = model(input_data)
        output = output.detach().numpy()
        index = np.argmax(output)
        print(f"Prediction successful, index: {index}")
        return index
    except Exception as e:
        print(f"Error in prediction function: {e}")
        raise


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        try:
            image = request.files['image']
            filename = image.filename
            file_path = os.path.join('static/uploads', filename)
            print(f"Saving image to {file_path}")
            image.save(file_path)
            print(f"Image saved to {file_path}")
            
            pred = prediction(file_path)
            print(f"Prediction result: {pred}")
            
            title = disease_info['disease_name'][pred]
            description = disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]
            
            print("Successfully processed prediction and retrieved data")
            
            return render_template('submit.html', title=title, desc=description, prevent=prevent, 
                                  image_url=image_url, pred=pred, sname=supplement_name, 
                                  simage=supplement_image_url, buy_link=supplement_buy_link)
        except Exception as e:
            print(f"Error in submit route: {e}")
            return f"An error occurred: {str(e)}"

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), 
                           disease=list(disease_info['disease_name']), 
                           buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
