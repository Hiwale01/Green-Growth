import os
import uuid
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for flash messages

# Create directory for uploads
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load disease information from CSV
def load_disease_data():
    try:
        df = pd.read_csv("updated_supplement_info.csv", encoding="latin1")
        df["Disease_name"] = df["Disease_name"].str.strip().str.lower()  # Normalize disease names
        print("✅ Disease data loaded successfully.")
        return df
    except Exception as e:
        print(f"❌ Error loading disease data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame if error

disease_data = load_disease_data()

# Load TensorFlow Model
def load_model():
    model_path = "trained_plant_disease_model.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file '{model_path}' not found.")
    
    print("✅ Loading TensorFlow model...")
    return tf.keras.models.load_model(model_path)

try:
    model = load_model()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None  # Set to None if model loading fails

# Disease class labels
class_names = [
    'apple : scab', 'apple : black rot', 'apple : cedar rust', 'apple : healthy',
    'blueberry : healthy', 'cherry : powdery mildew', 'cherry : healthy',
    'corn_(maize)___cercospora_leaf_spot_gray_leaf_spot', 'corn_(maize)___common_rust',
    'corn_(maize)___northern_leaf_blight', 'corn_(maize)___healthy',
    'grape___black_rot', 'grape___esca_(black_measles)', 'grape___leaf_blight',
    'grape___healthy', 'orange___huanglongbing', 'peach___bacterial_spot',
    'peach___healthy', 'pepper,_bell___bacterial_spot', 'pepper,_bell___healthy',
    'potato___early_blight', 'potato___late_blight', 'potato___healthy',
    'raspberry___healthy', 'soybean___healthy', 'squash___powdery_mildew',
    'strawberry___leaf_scorch', 'strawberry___healthy', 'tomato___bacterial_spot',
    'tomato___early_blight', 'tomato___late_blight', 'tomato___leaf_mold',
    'tomato___septoria_leaf_spot', 'tomato___spider_mites_two-spotted_spider_mite',
    'tomato___target_spot', 'tomato___yellow_leaf_curl_virus', 'tomato___mosaic_virus',
    'tomato___healthy'
]

# Model Prediction Function
def model_prediction(image_path):
    try:
        if model is None:
            print("❌ Model is not loaded!")
            return None

        print(f"🔍 Processing image: {image_path}")
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)  # Convert to batch
        
        predictions = model.predict(input_arr)
        result_index = np.argmax(predictions)

        print(f"✅ Prediction complete. Class Index: {result_index}, Class Name: {class_names[result_index]}")
        return result_index
    except Exception as e:
        print(f"❌ Error in model prediction: {str(e)}")
        return None

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    print("🔍 Received request at /predict")

    if "file" not in request.files:
        print("❌ No file part in request")
        flash("No file part!", "danger")
        return redirect(url_for("home"))

    file = request.files["file"]
    print(f"📂 Received file: {file.filename}")

    if file.filename == "":
        print("❌ No file selected")
        flash("No selected file!", "danger")
        return redirect(url_for("home"))

    try:
        # Verify file type (ensure it's an image)
        if not file.mimetype.startswith("image"):
            print("❌ Uploaded file is not an image!")
            flash("Only image files are allowed!", "danger")
            return redirect(url_for("home"))

        # Save file
        filename = str(uuid.uuid4()) + "_" + file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        print(f"✅ File saved: {filepath}")

        # Predict Disease
        result_index = model_prediction(filepath)
        if result_index is None:
            print("❌ Error in image processing or prediction.")
            flash("Error in image processing. Please try again.", "danger")
            return redirect(url_for("home"))

        predicted_disease = class_names[result_index].strip().lower()  # Normalize predicted disease name
        print(f"✅ Predicted disease: {predicted_disease}")

        # Retrieve disease information
        disease_info = disease_data[disease_data["Disease_name"] == predicted_disease]

        if disease_info.empty:
            print("❌ No matching disease found in database.")
            flash("No information found for the predicted disease.", "warning")
            return render_template("result.html", image_path=filepath, disease=predicted_disease, no_data=True)

        # Extract disease details
        supplement_name = disease_info.iloc[0]["supplement name"]
        supplement_image = disease_info.iloc[0]["supplement image"]
        buy_link = disease_info.iloc[0]["buy link"]
        description = disease_info.iloc[0]["description"]
        possible_steps = disease_info.iloc[0]["Possible Steps"]
        disease_reference_image = disease_info.iloc[0]["image_url"]

        return render_template("result.html", image_path=filepath, disease=predicted_disease,
                               supplement_name=supplement_name, supplement_image=supplement_image,
                               buy_link=buy_link, description=description, possible_steps=possible_steps,
                               disease_reference_image=disease_reference_image, no_data=False)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        flash(f"Error: {str(e)}", "danger")
        return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
