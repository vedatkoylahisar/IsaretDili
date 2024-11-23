import cv2
import mediapipe as mp

camera = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = camera.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    hlms = hands.process(imgRGB)
    
    if hlms.multi_hand_landmarks:
        for handlandmarks in hlms.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handlandmarks, mpHands.HAND_CONNECTIONS)
    
    cv2.imshow("Camera", img)

    # 'q' tuþuna basýldýðýnda döngüyü kýr
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera ve pencereleri düzgünce kapat
camera.release()
cv2.destroyAllWindows()
