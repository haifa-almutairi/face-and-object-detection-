import cv2
from ultralytics import YOLO


# INITIALIZATION

# 1. Load face detection model (Haar Cascade)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Load YOLO object detection model (version 11)
yolo_model = YOLO("yolo/yolo11n.pt")

# 3. Start webcam
camera = cv2.VideoCapture(0)
print("Running Combined Detection. Press 'q' to quit.")


# MAIN LOOP

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # flips the camera to make it look natural
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert the frame to grayscale

    # 1. Try Face Detection first
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0: # if faces are NOT empty (which means faces are detected)
        # Draw face rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) # Red box
        cv2.putText(frame, "Face Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        # 2. If no face detected, run YOLO object detection
        results = yolo_model(frame, verbose=False)

        for r in results:
            boxes = r.boxes
            classes = r.names
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0] # confidence score
                cls_id = int(box.cls[0]) # class id
                label = classes[cls_id] # converts class id to actual object names

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2) # Green box
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Label + Confidence Score

        cv2.putText(frame, "Object Detection Active", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 3. Show result
    cv2.imshow("Face and Object Detection", frame)

    # Quit condition when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()
