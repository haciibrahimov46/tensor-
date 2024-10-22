import cv2
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

prev_direction = ""
smile_detected = False

start_smile_time = None
start_direction_time = None

required_duration = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        if len(eyes) >= 2:
            eye1 = eyes[0]
            eye2 = eyes[1]

            eye1_center = (eye1[0] + eye1[2] // 2, eye1[1] + eye1[3] // 2)
            eye2_center = (eye2[0] + eye2[2] // 2, eye2[1] + eye2[3] // 2)

            eye_delta_x = eye2_center[0] - eye1_center[0]

            current_time = time.time()

            if eye_delta_x > 20:  
                if prev_direction != "sağa döndü":
                    start_direction_time = current_time
                    prev_direction = "sağa döndü"
                elif start_direction_time and current_time - start_direction_time >= required_duration:
                    print("Sağa döndü")
                    start_direction_time = None  
            elif eye_delta_x < -20:  
                if prev_direction != "sola döndü":
                    start_direction_time = current_time
                    prev_direction = "sola döndü"
                elif start_direction_time and current_time - start_direction_time >= required_duration:
                    print("Sola döndü")
                    start_direction_time = None 
            else:
                prev_direction = ""
                start_direction_time = None

        
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        if len(smiles) > 0: 
            if not smile_detected:
                start_smile_time = current_time  
                smile_detected = True
            elif start_smile_time and current_time - start_smile_time >= required_duration: 
                print("Gülümsedi")
                smile_detected = False
                start_smile_time = None
        else:
            smile_detected = False
            start_smile_time = None

    cv2.imshow('Yüz ve Gülümseme Tespiti', frame)

    if cv2.waitKey(1) & 0xFF == ord('k'):
        break

cap.release()
cv2.destroyAllWindows()
