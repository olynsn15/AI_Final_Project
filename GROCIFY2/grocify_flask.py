import cv2
import supervision as sv
from ultralytics import YOLO
from flask import Flask, render_template, Response, request, jsonify
import logging
from collections import defaultdict

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

model = YOLO('best_train4.pt')
logging.info("YOLO model loaded successfully")

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

cap = None
is_webcam_on = False

stock_thresholds = {
    "Apple": 3,
    "Banana": 4,
    "Beetroot": 4,
    "Bell pepper": 3,
    "Black pepper": 2,
    "Bread": 1,
    "Broccoli": 3,
    "Bun": 2,
    "Butter": 2,
    "Cabbage": 3,
    "Cardamom": 2,
    "Carrot": 3,
    "Catfish": 2,
    "Cauliflower": 3,
    "Cheese": 4,
    "Chicken": 2,
    "Chilli Pepper": 2,
    "Cinnamon": 1,
    "Coriander": 2,
    "Corn": 3,
    "Crawfish": 2,
    "Cucumber": 2,
    "Egg": 6,
    "Eggplant": 4,
    "Garlic": 4,
    "Ginger": 2,
    "Grapes": 2,
    "Jalepeno": 3,
    "Kiwi": 4,
    "Lettuce": 3,
    "Lobster": 2,
    "Mackerel": 2,
    "Mango": 3,
    "Milk": 2,
    "Olive": 5,
    "Onion": 6,
    "Orange": 6,
    "Paprika": 3,
    "Pear": 5,
    "Peas": 6,
    "Pineapple": 3,
    "Pistachio": 2,
    "Pomegranate": 4,
    "Pork": 3,
    "Potato": 8,
    "Radish": 5,
    "Rosemary": 2,
    "Soya Beans": 4,
    "Spring onion": 4,
    "Strawberry": 6,
    "Thyme": 2,
    "Tomato": 6,
    "Walnut": 2,
    "Watermelon": 2,
    "clove": 2,
    "duck": 2,
    "lemon": 4,
    "paer": 5,
    "raddish": 5,
    "spinach": 3,
    "sweet potato": 4
}

def generate_frames():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Unable to read camera feed")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model.predict(frame, conf=0.25)[0]
        detections = sv.Detections.from_ultralytics(result)

        item_counts = defaultdict(int)
        for i in range(len(detections.xyxy)):
            class_id = int(result.boxes.cls[i].item())  
            class_name = result.names[class_id]  
            item_counts[class_name] += 1

        annotated_image = box_annotator.annotate(scene=frame, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        y_offset = 30  
        for item, count in item_counts.items():
            if item in stock_thresholds and count < stock_thresholds[item]:
                warning_text = f"Low stock: {item} ({count}/{stock_thresholds[item]})"
                cv2.putText(
                    annotated_image,
                    warning_text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
                y_offset += 30 

        ret, buffer = cv2.imencode('.jpg', annotated_image)
        if not ret:
            break

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    global is_webcam_on
    if is_webcam_on:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return jsonify({"error": "Webcam is off."})

@app.route('/')
def index():
    return render_template('grocify.html')

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global is_webcam_on
    is_webcam_on = True
    return jsonify({"status": "Webcam started"})

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global is_webcam_on, cap
    is_webcam_on = False
    if cap is not None:
        cap.release()
    return jsonify({"status": "Webcam stopped"})

if __name__ == '__main__':
    app.run(debug=True)
