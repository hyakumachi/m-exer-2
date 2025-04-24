import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

model = load_model('fruit_recognition_model.h5')

with open('class_indices.json', 'r') as f:
    fruit_classes = json.load(f)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    resized = cv2.resize(frame, (100, 100))
    normalized = resized / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)

    
    predictions = model.predict(input_tensor, verbose=0)
    print("Raw Predictions:", predictions)

    
    class_idx = np.argmax(predictions)
    confidence = np.max(predictions)
    label = f"{fruit_classes[class_idx]} ({confidence:.2f})"


    
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
    cv2.imshow('Fruit Recognition', frame)

                
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
