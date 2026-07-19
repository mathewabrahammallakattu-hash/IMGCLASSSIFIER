import os
import base64
import io
import tensorflow as tf
from flask import Flask, request, render_template_string

# 1. Initialize Flask App
app = Flask(__name__)

# Locate the model file in the current working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cat_dog_classifier.keras")

# 2. Global variable to store loaded model safely
model = None
model_status = "Loading..."

try:
    if os.path.exists(MODEL_PATH):
        # We load without compiling because we only need it for inference/prediction
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model_status = "Ready"
        print("Successfully loaded cat_dog_classifier.keras!")
    else:
        model_status = "Error: cat_dog_classifier.keras file not found in this folder."
        print(model_status)
except Exception as e:
    model_status = f"Error loading model: {str(e)}"
    print(model_status)


# 3. Modern, Self-Contained Web UI (Dark Mode)
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classifier Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            background: #0f172a; 
            color: #f8fafc; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            min-height: 100vh; 
            padding: 20px;
        }
        .container { 
            background: #1e293b; 
            border-radius: 16px; 
            padding: 40px; 
            width: 100%; 
            max-width: 500px; 
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            text-align: center;
            border: 1px solid #334155;
        }
        h2 { margin-bottom: 8px; font-size: 24px; color: #f1f5f9; }
        .subtitle { color: #94a3b8; font-size: 14px; margin-bottom: 25px; }
        
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 99px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 20px;
            background: {% if status == 'Ready' %}#16a34a{% else %}#dc2626{% endif %};
        }

        .upload-area {
            border: 2px dashed #475569;
            border-radius: 12px;
            padding: 30px 20px;
            cursor: pointer;
            margin-bottom: 20px;
            transition: all 0.2s;
        }
        .upload-area:hover { border-color: #6366f1; background: #334155; }
        input[type="file"] { display: none; }
        .upload-label { cursor: pointer; color: #cbd5e1; display: block; font-weight: 500; }

        .btn { 
            background: linear-gradient(to right, #4f46e5, #7c3aed); 
            color: white; 
            border: none; 
            width: 100%; 
            padding: 14px; 
            border-radius: 8px; 
            font-size: 16px; 
            font-weight: bold; 
            cursor: pointer;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .btn:hover { opacity: 0.9; }
        .btn:disabled { background: #475569; cursor: not-allowed; }

        .result-panel { 
            margin-top: 30px; 
            padding-top: 20px; 
            border-top: 1px solid #334155; 
        }
        .preview { 
            max-width: 100%; 
            max-height: 200px; 
            border-radius: 8px; 
            margin-bottom: 15px;
            border: 2px solid #475569;
        }
        .prediction-text { font-size: 26px; font-weight: bold; color: #38bdf8; }
        .val-text { font-size: 13px; color: #94a3b8; margin-top: 5px; font-family: monospace; }
    </style>
</head>
<body>

<div class="container">
    <h2>Cat vs Dog Scanner</h2>
    <div class="subtitle">Deep Learning Classifier UI</div>
    
    <div class="status-badge">Model Status: {{ status }}</div>

    <form action="/" method="POST" enctype="multipart/form-data">
        <div class="upload-area" onclick="document.getElementById('file-input').click()">
            <span class="upload-label" id="file-name">📁 Click to select an image...</span>
            <input type="file" id="file-input" name="user_image" accept="image/*" required>
        </div>
        <button type="submit" class="btn" {% if status != 'Ready' %}disabled{% endif %}>Run Classification</button>
    </form>

    {% if img_data %}
        <div class="result-panel">
            <img src="data:image/jpeg;base64,{{ img_data }}" class="preview" alt="User Image">
            <div class="prediction-text">{{ final_label }}</div>
            <div class="val-text">Raw Output: {{ raw_val }}</div>
        </div>
    {% endif %}
</div>

<script>
    // Changes the file picker placeholder string once an image file is chosen
    document.getElementById('file-input').onchange = function() {
        var name = this.files[0] ? this.files[0].name : "Click to select an image...";
        document.getElementById('file-name').textContent = name;
    };
</script>

</body>
</html>
"""

# 4. Routing Controller
@app.route("/", methods=["GET", "POST"])
def index():
    context = {"status": model_status}

    if request.method == "POST":
        file = request.files.get("user_image")
        if model and file and file.filename != "":
            try:
                # Read bytes from request without saving anything to the storage drive
                file_bytes = file.read()
                
                # Run exact image preprocessing steps from your script using in-memory byte streams
                img = tf.keras.utils.load_img(io.BytesIO(file_bytes), target_size=(224, 224))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)
                
                # Execute prediction pipeline matching your code
                prediction = model.predict(img_array)
                raw_val = prediction[0][0]
                
                # Threshold condition checking from your original logic
                if raw_val > 0.5:
                    label = "Dog 🐶"
                else:
                    label = "Cat 🐱"
                
                # Encode file bytes to display the selection right on the web browser screen
                encoded_img = base64.b64encode(file_bytes).decode('utf-8')
                
                context.update({
                    "final_label": label,
                    "raw_val": round(float(raw_val), 5),
                    "img_data": encoded_img
                })
            except Exception as e:
                context["status"] = f"Processing Error: {str(e)}"

    return render_template_string(HTML_LAYOUT, **context)

if __name__ == "__main__":
    app.run(debug=True, port=5000)