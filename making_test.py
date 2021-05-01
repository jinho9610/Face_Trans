from keras.preprocessing.image import ImageDataGenerator
from mtcnn import MTCNN
import math
import cv2
import os

detector = MTCNN()

#targetX, targetY = 224, 224
targetX, targetY = 100, 100
angle_list = [-10, -5, -3, 3, 5, 10]


def isValidFace(face):
    if len(detector.detect_faces(face)) == 1:
        return True
    else:
        return False


def TestDatasetMaker(root_dir):
    for i, img_path in enumerate(os.listdir(root_dir)):
        abs_img_path = os.path.join(root_dir, img_path)
        img = cv2.cvtColor(cv2.imread(abs_img_path), cv2.COLOR_BGR2RGB)
        face_info = detector.detect_faces(img)[0]  # 얼굴 검출

        box = face_info['box']
        face_center_x = box[0] + int(1/2 * box[2])
        face_center_y = box[1] + int(1/2 * box[3])
        a = int(1/2 * box[2]) if box[2] > box[3] else int(1/2 * box[3])
        box = face_info['box']  # 바운더리 박스
        # 최대한 얼굴 형태 유지하며 정방형으로 잘라냄
        face = img[face_center_y -
                   a:face_center_y + a, face_center_x - a:face_center_x + a]
        # 얼굴 (224, 224)로 만듦
        face = cv2.resize(face, dsize=(targetX, targetY))

        isFace = isValidFace(face)
        if isFace:
            new_path = 'test/test_face' + str(i + 1) + '.jpg'
            print(new_path)
            cv2.imwrite(new_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        else:
            print(img_path + 'is not Face!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('==================================================')


if __name__ == '__main__':
    TestDatasetMaker('test_photos')
