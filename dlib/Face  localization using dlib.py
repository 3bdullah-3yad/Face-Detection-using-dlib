import dlib
import cv2

detector = dlib.get_frontal_face_detector()

model_path= r"D:\Projects\Computer Vision\Face detection using dlib\shape_predictor_68_face_landmarks.dat"
predictor= dlib.shape_predictor(model_path)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray_frame)
    
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        landmarks = predictor(gray_frame, face)
        for lm in range(68):
            x = landmarks.part(lm).x
            y = landmarks.part(lm).y
            
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        
        
    cv2.imshow("Ayad", cv2.flip(frame, 1))
    
    
    if cv2.waitKey(1) == ord("a"):
        break

cv2.destroyAllWindows()