import cv2
import base64

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if ret:
    cnt = cv2.imencode('.jpg', frame)[1]
    b64 = base64.encodestring(cnt)
    imagedata = base64.
    print b64
