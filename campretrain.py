# Importing libraries
import cv2
import numpy as np
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import time
import threading
import queue

# Dowloading Pretrained weights
model = VisionEncoderDecoderModel.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)
feature_extractor = ViTImageProcessor.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Moving model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Config
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


# Prediction function
def predict_step(image):
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return preds[0].strip() if preds else None


# Multithreading for running captioning seperately
def caption_thread(frame_queue, caption_queue, stop_event):
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:
                break
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            caption = predict_step(pil_image)
            caption_queue.put(caption)
            frame_queue.task_done()
        except queue.Empty:
            continue


# Opening Webcam and making new thread
cap = cv2.VideoCapture(0)
frame_queue = queue.Queue(maxsize=1)
caption_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

thread = threading.Thread(
    target=caption_thread, args=(frame_queue, caption_queue, stop_event), daemon=True
)
thread.start()

last_caption_time = time.time()
caption = ""

# Running model on webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_queue.empty():
        frame_queue.put(frame)

    # Mirror the frame
    frame = cv2.flip(frame, 1)

    current_time = time.time()
    if current_time - last_caption_time >= 1:
        last_caption_time = current_time
        try:
            caption = caption_queue.get_nowait()
        except queue.Empty:
            pass

    if caption:
        (w, h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        x, y = 10, frame.shape[0] - 10
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (x, y - h - 10), (x + w + 10, y + 10), (0, 255, 0), -1
        )
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(
            frame,
            caption,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imshow("Webcam Captioning", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        stop_event.set()
        break

frame_queue.put(None)
thread.join()
cap.release()
cv2.destroyAllWindows()
