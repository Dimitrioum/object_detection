import cv2
import dlib
import numpy as np
import pickle
from operator import itemgetter
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import datetime
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/malov/PycharmProjects/object_detection/shape_predictor_68_face_landmarks.dat')
recognizer = dlib.face_recognition_model_v1('/home/malov/PycharmProjects/object_detection/dlib_face_recognition_resnet_model_v1.dat')
directory = '/home/malov/PycharmProjects/object_represent/generated-embeddings'



TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
OUTER_EYES_AND_NOSE = [36, 45, 33]

def _get_aligned_frame(frame):
    detection = detector(frame, 1)
    bb = max(detection, key=lambda rect: rect.width() * rect.height())
    points = predictor(frame, bb)
    landmarks = list(map(lambda p: (p.x, p.y), points.parts()))
    np_landmarks = np.float32(landmarks)
    np_landmarks_indices = np.array(INNER_EYES_AND_BOTTOM_LIP)
    affine_transform = cv2.getAffineTransform(np_landmarks[np_landmarks_indices], 96 * MINMAX_TEMPLATE[np_landmarks_indices])
    return cv2.warpAffine(frame, affine_transform, (96, 96)) # type: np.array, like a frame


def _camera_init(index):
    text = 'malov'
    font = cv2.FONT_HERSHEY_SIMPLEX
    faceCascade = cv2.CascadeClassifier('/home/malov/PycharmProjects/object_detection/cascade_face_detection.xml')
    cap = cv2.VideoCapture(index)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Terminal1', frame) if ret else 'Camera is not active, choose another terminal'
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

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, text, (int((x + w) / 2), int((y + h) / 2)), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
            aligned_frame = frame[y:(y + w), x:(x + h)]

    cap.release()
    cv2.destroyAllWindows()


def _train(reps):
    zero_user = np.zeros((10, 128))
    ones_user = np.ones((10, 128))
    X = np.concatenate((reps, zero_user))
    X = np.concatenate((X, ones_user))
    y = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,3,3])

    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    print(clf.score(X, y))
    return clf

    # param_grid = [
    #     {'C': [1, 10, 100, 1000],
    #      'kernel': ['linear']},
    #     {'C': [1, 10, 100, 1000],
    #      'gamma': [0.001, 0.0001],
    #      'kernel': ['rbf']}
    # ]

    # X_pca = PCA(n_components=50).fit_transform(X, X)
    # tsne = TSNE(n_components=2, init='random', random_state=0)
    # X_r = tsne.fit_transform(X_pca)

    # svm = GridSearchCV(SVC(C=1), param_grid, cv=5).fit(X, y)

    # labels_new = []
    # for i in range(10):
    #     labels_new.append(name)
    # labels = np.array(labels)
    # le = LabelEncoder().fit(labels)
    # labels_num = le.transform(labels)
    # clf = SVC(C=1, kernel='linear', probability=True)
    # clf.fit(reps, labels_num)
    # return le, clf

def _save_clf(le, clf):
    with open('/home/malov/PycharmProjects/object_detection/classifier.pkl', 'w') as raw_file:
        pickle.dump((le, clf), raw_file)

def _infer(index, clf):
    cap = cv2.VideoCapture(index)
    ret, frame = cap.read()
    reps = []
    if ret:
        frame1 = _get_aligned_frame(frame)
        win = dlib.image_window()
        win.clear_overlay()
        win.set_image(frame)
        dets = detector(frame, 1)
        for k, d in enumerate(dets):
            shape = predictor(frame, d)
            win.clear_overlay()
            win.add_overlay(d)
            win.add_overlay(shape)
            face_descriptor = recognizer.compute_face_descriptor(frame1, shape)
            face_info = np.array(face_descriptor)  # 128 characters
            reps.append(face_info)
        reps = np.array(reps)

    predictions = clf.predict_proba(reps)
    # maxI = np.argmax(predictions)
    # confidence = predictions[maxI]
    return predictions

def _decorator_print(function):
    def _decorator_pr(index, name):
        print(0)
        function(index, name)
        print(1)

@_decorator_print
def _user_photo(index, name):
    cap = cv2.VideoCapture(index)
    photos_list = []
    directory = '/home/malov/openface/{}'.format(name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    while len(photos_list) < 10:
        print(10-len(photos_list))
        ret, frame = cap.read()
        if ret:
            cv2.imshow('saving', frame)
            cv2.imwrite('{}/{}-{}-{} {}:{}:{}:{}.jpg'.format(directory,
                                                             datetime.datetime.now().year,
                                                             datetime.datetime.now().month,
                                                             datetime.datetime.now().day,
                                                             datetime.datetime.now().hour,
                                                             datetime.datetime.now().minute,
                                                             datetime.datetime.now().second,
                                                             datetime.datetime.now().microsecond), frame)

            photos_list.append(frame)
    cap.release()


def _get_reps(index):
    reps = []
    cap = cv2.VideoCapture(index)
    while len(reps) < 10:
        print(10 - len(reps))
        ret, frame = cap.read()
        if ret:
            frame1 = _get_aligned_frame(frame)
            win = dlib.image_window()
            win.clear_overlay()
            win.set_image(frame)
            dets = detector(frame, 1)
            for k, d in enumerate(dets):
                shape = predictor(frame, d)
                win.clear_overlay()
                win.add_overlay(d)
                win.add_overlay(shape)
                face_descriptor = recognizer.compute_face_descriptor(frame1, shape)
                face_info = np.array(face_descriptor) # 128 characters
                reps.append(face_info)
    cap.release()
    return np.array(reps)

def _add_user(name):
    new_label = []
    new_reps = _get_reps(0)
    old_labels = map(itemgetter(2), map(lambda x: x.split('/', pd.read_csv('/home/malov/openface/generated-embeddings/labels.csv').as_matrix()[:, 1])))
    old_reps = pd.read_csv('/home/malov/openface/generated-embeddings/reps.csv').as_matrix()
    for i in range(new_reps):
        new_label.append(name)
    new_labels = np.concatenate((old_labels, np.array(new_label)))
    new_reps = np.concatenate((old_reps, new_reps))


# def _save_photo(frame):
#     cv2.imwrite('dima.jpg', frame)







if __name__ == '__main__':
    # name = 'malov '
    # # reps = _get_reps(0)
    # # le, clf = _train(name, reps)
    # # _save_clf(le, clf)
    #
    # reps = _get_reps(0)
    # reps_vert = np.vstack(reps)
    # print(reps.shape, reps_vert.shape, reps_vert)
    # clf = _train(reps)
    # print(_infer(0, clf), )
    with open('/home/malov/PycharmProjects/FaceRecognition/Classifier/classifier.pkl', 'r') as f:
        (le, clf) = pickle.load(f)


























