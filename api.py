from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import json

app = Flask(__name__)

# Load YOLO model once
model_path = "best.pt"
model = YOLO(model_path)
names = model.names

def yolo_detection(frame):
    results_yolo = model.predict(frame)[0]
    for result in results_yolo.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if int(class_id) == 4:
            continue
        if score > 0.8:
            class_name = names[int(class_id)]
            return str(class_name)
    return None

def retrieve_workout_plan(name, filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        foods = data["foods"]

    for food in foods:
        if food['name'] == name:
            ret = {
                "data": [
                    {
                        "name": food['name'],
                        "calories_per_100g": food['calories_per_100g'],
                        "fats_per_100g": food['fats_per_100g'],
                        "carbohydrates_per_100g": food['carbohydrates_per_100g'],
                        "protein_per_100g": food['protein_per_100g'],
                        "fiber_per_100g": food['fiber_per_100g']
                    }
                ]
            }
            return ret
    return {"error": "No matching food item found."}

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    food_name = yolo_detection(frame)
    if not food_name:
        return jsonify({"error": "No valid detection"}), 400

    result = retrieve_workout_plan(food_name, filename="data.json")
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
