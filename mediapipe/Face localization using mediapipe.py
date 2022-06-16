import mediapipe as mp
import cv2

def show(img, name="Ayad"):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread(r"D:\Pics\1.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# show(img)

fm = mp.solutions.face_mesh.FaceMesh()

LandMarks = fm.process(img)

w, h, = img.shape[:-1]

for img_lm in LandMarks.multi_face_landmarks:
    for lm in range(468):
        pt = img_lm.landmark[lm]
        
        x = int(pt.x * w)
        y = int(pt.y * h)
        
        cv2.circle(img, (x, y), 2, (200, 100, 50), -1)
        
show(img)        
