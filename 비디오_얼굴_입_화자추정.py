from PIL import Image
from collections import deque
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
#import join_run as jr

IMG_SIZE = (34, 26)  # 입 이미지의 가로, 세로 사이즈
#categories = ["hyeontae", "jinho", "yoosung"]
categories = ["hyeontae", "jaehyeon", "jinho", "joohyeong", "yoosung"]
targetX, targetY = 100, 100

detector = MTCNN()

resnet = model_prediction.model
#model = load_model('models/2021_03_05_13_51_54.h5')
model = load_model('mouth_models/2021_04_02_01_57_41.h5')

class_participants = {}


def return_face(face_info, img):
    box = face_info['box']  # 바운더리 박스
    # x가 가로 방향, y가 세로 방향
    face_width, face_height = box[2], box[3]
    #print(face_width, face_height)
    face_center_x = box[0] + int(1/2 * box[2])
    face_center_y = box[1] + int(1/2 * box[3])

    # 얼굴의 가로, 세로 중 어느 길이에 맞추어 자를 것인지 결정하기 위한 인자
    a = int(1/2 * box[2]) if box[2] > box[3] else int(1/2 * box[3])

    try:  # 최대한 얼굴 형태 유지하며 정방형으로 잘라냄
        face = img[face_center_y -
                   a:face_center_y + a, face_center_x - a:face_center_x + a]
        # 얼굴 (224, 224)로 만듦

        #cv2.imshow('before', face)
        face = cv2.resize(face, dsize=(targetX, targetY))
        #face = jr.SR(Image.fromarray(face))
        #face = jr.SR(face)

        # print('after trans: ', type(face))
        #cv2.imshow('face', face)
        # cv2.waitKey(0)
        return face

    except:
        print('face is out of img range!!!!!')
        # print(face_info)
        return None


def contrast_check_mouth(ori, face_info, name):
    box = face_info['box']
    keypoints = face_info['keypoints']
    nose_x, nose_y = keypoints['nose'][0], keypoints['nose'][1]
    lm_x, lm_y = keypoints['mouth_left'][0], keypoints['mouth_left'][1]
    rm_x, rm_y = keypoints['mouth_right'][0], keypoints['mouth_right'][1]

    mouth = ori[nose_y + 6: box[1] + box[3], lm_x: rm_x]

    mouth_rect = [(lm_x, nose_y + 6), (rm_x, box[1] + box[3])]

    try:
        mouth = cv2.resize(mouth, dsize=(34, 26))

        ori = cv2.rectangle(
            ori, mouth_rect[0], mouth_rect[1], (255, 0, 0), 1)  # 입주변 박스 그리기

        mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)  # 입 사진 gray로 변경
        mouth_input = mouth.copy().reshape(
            (1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255
        pred = model.predict(mouth_input)

        pred = pred[0][0]

        if name is not 'UNKNOWN':
            if pred > 0.5:
                class_participants[name].append('o')
            else:
                class_participants[name].append('x')

            state = 'O %.1f' if pred > 0.5 else '- %.1f'
            state = state % pred

            if class_participants[name].count('o') > class_participants[name].count('x'):
                cv2.putText(ori, 'speaker', (mouth_rect[0][0] + int((mouth_rect[1][0] - mouth_rect[0][0]) / 4),
                                             mouth_rect[0][1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            else:
                cv2.putText(ori, state, (mouth_rect[0][0] + int((mouth_rect[1][0] - mouth_rect[0][0]) / 4),
                                         mouth_rect[0][1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    except:
        print('cannot detect mouth')
        pass

    return ori


def contrast_video_face_mouth_rec(input_video):
    length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(
        'output_videos/output_mouth.avi', fourcc, 29.97, (w, h))

    frame_num = 0
    while True:
        ret, img = input_video.read()

        if not ret:
            print("end of video.")
            break

        frame_num += 1
        # if frame_num < 180:
        #     continue

        print("Writing frame {} / {}".format(frame_num, length))
        img_for_detection = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_infos = detector.detect_faces(img_for_detection)  # 얼굴 정보 검출
        if len(face_infos) == 0:
            print('this img has no face!!!!!!!!!!!!!!!!!!!!!!!')

        else:
            for face_info in face_infos:
                # 인물 예측
                face = return_face(face_info, img)

                if face is not None:
                    name, acc = model_prediction.predict_by_model(Image.fromarray(
                        cv2.cvtColor(face, cv2.COLOR_BGR2RGB)), categories)

                    if class_participants.get(name, None) == None:
                        class_participants[name] = deque(
                            'x' * 15, maxlen=15)

                    box = face_info['box']
                    # 얼굴 바운더리 박스 그리기
                    img = cv2.rectangle(
                        img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 2)
                    # 예측 인물 이름 및 값 적기
                    if acc < 10:  # acc 너무 낮으면 UNKNOWN 처리
                        name = 'UNKNOWN'
                    img = cv2.putText(
                        img, name + ' '+str(acc), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    contrast_check_mouth(img, face_info, name)
                else:
                    print('NO FACE')
                    pass

        output_video.write(img)
        #cv2.imshow('img', img)
        # cv2.waitKey(0)

    input_video.release()


if __name__ == '__main__':
    input_video = cv2.VideoCapture('videos/INSTA.MP4')
    contrast_video_face_mouth_rec(input_video)
