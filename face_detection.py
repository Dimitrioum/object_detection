import cv2
import os
import subprocess as sp
import pickle
import openface
from PIL import Image
import imagehash
import dlib
import numpy as np
from sklearn.mixture import GMM


faceCascade = cv2.CascadeClassifier('/home/malov/PycharmProjects/object_detection/cascade_face_detection.xml')

cap = cv2.VideoCapture(0)

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
dlibFacePredictor = '/home/malov/PycharmProjects/object_detection/shape_predictor_68_face_landmarks.dat'
networkModel = '/home/malov/PycharmProjects/object_detection/nn4.small2.v1.t7'


# parser = argparse.ArgumentParser()
# parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
#                     default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
# parser.add_argument('--imgDim', type=int,
#                     help="Default image dimension.", default=96)
#
# args = parser.parse_args()
align = openface.AlignDlib(dlibFacePredictor)
net = openface.TorchNeuralNet(networkModel, imgDim=96, cuda=True)

identities = []

def getRep(frame):
    bb1 = align.getLargestFaceBoundingBox(frame)
    bbs = [bb1]
    if (len(bbs) == 0) or (bb1 is None):
        raise Exception("Unable to find a face")

    reps = []
    for bb in bbs:
        alignedFace = align.align(
            96,
            frame,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise Exception("Unable to align image")

        rep = net.forward(alignedFace)
        reps.append((bb.center().x, rep))
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps



def infer(frame):
    with open(r'/home/malov/openface/generated-embeddings/classifier.pkl', 'r') as f:
        (le, clf) = pickle.load(f)

    reps = getRep(frame)
    if len(reps) > 1:
        print("List of faces in image from left to right")
    for r in reps:
        rep = r[1].reshape(1, -1)
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]
        print("I think, you are {} with {:.2f} confidence.".format(person, confidence))

if __name__ == '__main__':
    text = 'malov'
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, frame = cap.read()
        if ret:

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, text, (int((x+w)/2), int((y+h)/2)), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
                aligned_frame = frame[y:(y+w), x:(x+h)]

                # infer(aligned_frame)
            #     cv2.imwrite('/home/malov/openface/cut_frame.jpg', frame[y:(y+w), x:(x+h)])
            cv2.imshow('Video', frame)
            # Display the resulting frame




# os.system('cd /home/malov/openface')

# result = sp.check_output("cd /home/malov/openface && ./demos/classifier.py infer ./generated-embeddings/classifier.pkl cut_frame.jpg", shell=True)


# print(result)
cap.release()
cv2.destroyAllWindows()
