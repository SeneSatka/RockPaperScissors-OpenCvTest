from tensorflow.keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
model = load_model('model.keras')
cap = cv2.VideoCapture(0)  
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_resized = cv2.resize(frame, (150, 150))    
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  

    prediction = model.predict(img_array)
    cv2.putText(frame, f"Paper: %{(prediction[0][0]*100):.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)    
    cv2.putText(frame, f"Rock: %{(prediction[0][1]*100):.2f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)    
    cv2.putText(frame, f"Scissors: %{(prediction[0][2]*100):.2f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 100), 1)    
    cv2.imshow('Real-time Prediction', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
