import gradio as gr
import cv2
from ultralytics import YOLO

model = YOLO('best_train4.pt')

def detect_objects(image, confidence):
    results = model.predict(source=image, conf=confidence)
    
    annotated_image = image.copy()
    detected_objects = []
    for result in results:
        for box in result.boxes.xyxy.tolist():  # Get bounding box coordinates
            class_id = int(result.boxes.cls.tolist()[0])
            class_name = result.names[class_id]
            confidence_score = float(result.boxes.conf.tolist()[0])

            # Add detection to the list
            detected_objects.append(f"{class_name}: {confidence_score:.2f}")

            # Draw bounding box
            start_point = tuple(map(int, box[:2]))
            end_point = tuple(map(int, box[2:]))
            cv2.rectangle(annotated_image, start_point, end_point, (0, 255, 0), 2)

            # Annotate with class name
            cv2.putText(annotated_image, class_name, start_point, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated_image, "\n".join(detected_objects)

with gr.Blocks() as demo:
    gr.Markdown("## 🥕 Grocify")
    gr.Markdown("Upload gambar untuk mendeteksi sayur dan buah!")

    with gr.Row():
        # Image input
        image_input = gr.Image(label="Upload Gambar", type="numpy")
        confidence_slider = gr.Slider(0.1, 1.0, step=0.1, value=0.5, label="Confidence Threshold")

    detect_button = gr.Button("Deteksi Objek")

    with gr.Row():
        output_image = gr.Image(label="Hasil Deteksi")
        output_text = gr.Textbox(label="Daftar Objek Terdeteksi", lines=5)

    gr.Markdown("### Petunjuk Penggunaan:\n1. Upload gambar yang berisi sayur atau buah.\n2. Sesuaikan Confidence Threshold jika diperlukan.\n3. Klik tombol Deteksi Objek untuk melihat hasilnya.")

    # Define interaction
    detect_button.click(fn=detect_objects,
                        inputs=[image_input, confidence_slider],
                        outputs=[output_image, output_text])

# Launch the Gradio app
demo.launch()
