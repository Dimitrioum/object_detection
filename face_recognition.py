
import cv2


if __name__ == '__main__':
    frame = cv2.imread('/home/malov/frame.jpg')
    faceCascade = cv2.CascadeClassifier('/home/malov/PycharmProjects/object_detection/cascade_face_detection.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)


