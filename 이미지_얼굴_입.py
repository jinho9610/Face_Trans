from PIL import Image
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model, load_model
from functools import partial
from inception_resnet_v1_lcl import *
from mtcnn import MTCNN
import matplotlib.image as mpimg
import model_prediction
import numpy as np
import cv2
import os

IMG_SIZE = (34, 26)  # 입 이미지의 가로, 세로 사이즈
categories = ["hyeontae", "jinho", "yoosung"]
targetX, targetY = 224, 224

detector = MTCNN()

resnet = model_prediction.model
model = load_model('models/2021_03_05_13_51_54.h5')


def return_face(face_info, img):
    box = face_info['box']  # 바운더리 박스
    # x가 가로 방향, y가 세로 방향
    face_width, face_height = box[2], box[3]
    face_center_x = box[0] + int(1/2 * box[2])
    face_center_y = box[1] + int(1/2 * box[3])

    # 얼굴의 가로, 세로 중 어느 길이에 맞추어 자를 것인지 결정하기 위한 인자
    a = int(1/2 * box[2]) if box[2] > box[3] else int(1/2 * box[3])

    # 최대한 얼굴 형태 유지하며 정방형으로 잘라냄
    face = img[face_center_y -
               a:face_center_y + a, face_center_x - a:face_center_x + a]
    # 얼굴 (224, 224)로 만듦
    face = cv2.resize(face, dsize=(targetX, targetY))

    return face


def check_mouth(ori, face_info):
    box = face_info['box']
    keypoints = face_info['keypoints']
    nose_x, nose_y = keypoints['nose'][0], keypoints['nose'][1]
    lm_x, lm_y = keypoints['mouth_left'][0], keypoints['mouth_left'][1]
    rm_x, rm_y = keypoints['mouth_right'][0], keypoints['mouth_right'][1]

    mouth = ori[nose_y + 6: box[1] + box[3], lm_x: rm_x]

    print(nose_y + 6, box[1] + box[3])
    print(lm_x, rm_x)

    mouth_rect = [(lm_x, nose_y + 6), (rm_x, box[1] + box[3])]

    mouth = cv2.resize(mouth, dsize=(34, 26))
    ori = cv2.rectangle(
        ori, mouth_rect[0], mouth_rect[1], (255, 0, 0), 1)  # 입주변 박스 그리기

    mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)  # 입 사진 gray로 변경
    mouth_input = mouth.copy().reshape(
        (1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255
    pred = model.predict(mouth_input)
    state = 'O %.1f' if pred > 0.1 else '- %.1f'
    state = state % pred

    cv2.putText(ori, state, (mouth_rect[0][0] + int((mouth_rect[1][0] - mouth_rect[0][0]) / 4),
                             mouth_rect[0][1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return ori


def img_face_mouth_rec(img_path):
    img = cv2.imread(img_path)
    img_for_detection = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_infos = detector.detect_faces(img_for_detection)  # 얼굴 정보 검출
    if len(face_infos) == 0:
        print('this img has no face!!!!!!!!!!!!!!!!!!!!!!!')

    else:
        for face_info in face_infos:
            # 인물 예측
            face = return_face(face_info, img)
            name, acc = model_prediction.predict_by_model(Image.fromarray(
                cv2.cvtColor(face, cv2.COLOR_BGR2RGB)), categories)
            box = face_info['box']
            # 얼굴 바운더리 박스 그리기
            img = cv2.rectangle(
                img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 2)
            print(name, acc)
            # 예측 인물 이름 및 값 적기
            if acc < 50:  # acc 너무 낮으면 UNKNOWN 처리
                name = 'UNKNOWN'
            img = cv2.putText(
                img, name + ' '+str(acc), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            check_mouth(img, face_info)

    cv2.imshow('img', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    p = 'not_face/33.jpg'
    img_face_mouth_rec(p)
