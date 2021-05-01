from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import cv2
import math
import os

mtcnn = MTCNN(image_size=240, margin=0, keep_all=True,
              min_face_size=40)  # keep_all=True


def face_align(img_path):
    img = Image.open(img_path)

    try:
        img_cropped_list, prob_list = mtcnn(img, return_prob=True)
        if img_cropped_list is not None:
            boxes, _, faces = mtcnn.detect(
                img, landmarks=True)  # detect는 PIL 상태에서 진행
            box = boxes[0]
            face = faces[0]

            img = np.array(img)
            left_eye_x, left_eye_y = int(face[0][0]), int(face[0][1])
            right_eye_x, right_eye_y = int(face[1][0]), int(face[1][1])

            theta = math.degrees(
                math.atan(-(left_eye_y-right_eye_y) / (right_eye_x - left_eye_x)))

            h, w, c = img.shape

            matrix = cv2.getRotationMatrix2D((w/2, h/2), theta, 1)
            img = cv2.warpAffine(img, matrix, (w, h))
            #img = Image.fromarray(img)
            # img_path = 'warped_' + img_path
            pic_name = img_path.split('\\')[1]
            pic_name = 'warped_' + pic_name
            print(img_path.split('\\')[0] + '\\' + pic_name)
            # print(img_path.slice('/')[-1])
            os.remove(img_path)
            #img.save(img_path.split('\\')[0] + '\\' + pic_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow('img', img)
            cv2.waitKey(10)
            cv2.imwrite(img_path.split('\\')[0] + '\\' + pic_name, img)

    except:
        print('---------------------------failed--------------------------')
        pass


if __name__ == '__main__':
    #dir_path = 'jinho'
    dir_path = input('누구?: ')

    for r_img_path in os.listdir(dir_path):
        a_img_path = os.path.join(dir_path, r_img_path)

        face_align(a_img_path)
