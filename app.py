import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load trained model
model = load_model("trained.h5")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def is_xray(img_array):
    """Check if image is likely a chest X-ray (grayscale/low color variance)."""
    # Convert to 0-255 range
    img = img_array[0] * 255.0

    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # X-rays have very similar R, G, B channels (near grayscale)
    rg_diff = np.mean(np.abs(r.astype(float) - g.astype(float)))
    rb_diff = np.mean(np.abs(r.astype(float) - b.astype(float)))
    gb_diff = np.mean(np.abs(g.astype(float) - b.astype(float)))

    avg_diff = (rg_diff + rb_diff + gb_diff) / 3

    # If average color channel difference is high, it's a colorful image (not X-ray)
    return avg_diff < 10  # threshold — X-rays are near-grayscale

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Preprocess image (300x300)
            img = image.load_img(filepath, target_size=(300, 300))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Check if image is an X-ray
            if not is_xray(img_array):
                prediction = "Unknown"
            else:
                # Predict
                result = model.predict(img_array)

                # Handle both sigmoid & softmax
                if result.shape[1] == 1:  # sigmoid
                    confidence = result[0][0]
                    if confidence > 0.75:
                        prediction = "Pneumonia"
                    elif confidence < 0.25:
                        prediction = "Normal"
                    else:
                        prediction = "Unknown"
                else:  # softmax
                    confidence = np.max(result)
                    pred = np.argmax(result)
                    if confidence >= 0.75:
                        if pred == 1:
                            prediction = "Pneumonia"
                        else:
                            prediction = "Normal"
                    else:
                        prediction = "Unknown"

            image_path = filepath

    return render_template('ind.html', prediction=prediction, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
    