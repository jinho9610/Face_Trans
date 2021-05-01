from keras.preprocessing.image import ImageDataGenerator
from mtcnn import MTCNN
import math
import cv2
import os

detector = MTCNN()

#targetX, targetY = 224, 224
targetX, targetY = 80, 80
angle_list = [-10, -5, -3, 3, 5, 10]


def isValidFace(face):
    if len(detector.detect_faces(face)) == 1:
        return True
    else:
        return False


def FaceDatasetMaker(root_dir):
    for mode_dir in os.listdir(root_dir):  # mode_dir : train, val
        # name_dir : hyeontae, jinho, yoosung
        abs_mode_dir = os.path.join(root_dir, mode_dir)
        for name_dir in os.listdir(abs_mode_dir):
            abs_name_dir = os.path.join(abs_mode_dir, name_dir)
            for i, img_path in enumerate(os.listdir(abs_name_dir)):
                abs_img_path = os.path.join(abs_name_dir, img_path)
                img = cv2.cvtColor(cv2.imread(abs_img_path), cv2.COLOR_BGR2RGB)
                face_info = detector.detect_faces(img)[0]  # 얼굴 검출

                left_eye_x = face_info['keypoints']['left_eye'][0]  # 왼쪽 눈 x
                left_eye_y = face_info['keypoints']['left_eye'][1]  # 왼쪽 눈 y
                right_eye_x = face_info['keypoints']['right_eye'][0]  # 오른쪽 눈 x
                right_eye_y = face_info['keypoints']['right_eye'][1]  # 오른쪽 눈 y

                # 눈 수평을 맞추기 위한 눈 각도 계산
                theta = math.degrees(
                    math.atan(-(left_eye_y-right_eye_y) / (right_eye_x - left_eye_x)))
                h, w, c = img.shape
                matrix = cv2.getRotationMatrix2D((w/2, h/2), theta, 1)
                img = cv2.warpAffine(img, matrix, (w, h))

                face_info = detector.detect_faces(img)[0]  # 수평 맞춘 얼굴에 대해 재검출
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

                isFace = isValidFace(face)
                if isFace:
                    # 이제 (224, 224)로 잘 만들어진 얼굴 저장
                    new_path = 'data/' + mode_dir + '/' + name_dir + \
                        '/' + name_dir + str(i + 1) + '.jpg'

                    if not os.path.exists('data/' + mode_dir + '/' + name_dir):
                        os.mkdir('data/' + mode_dir + '/' + name_dir)
                    print(new_path)
                    cv2.imwrite(new_path, cv2.cvtColor(
                        face, cv2.COLOR_RGB2BGR))
                else:
                    print(img_path + 'is not Face!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('==================================================')


def DataAugmenter(root_dir):
    for mode_dir in os.listdir(root_dir):  # mode_dir : train, val
        abs_mode_dir = os.path.join(root_dir, mode_dir)
        # name_dir : hyeontae, jinho, yoosung
        for name_dir in os.listdir(abs_mode_dir):
            abs_name_dir = os.path.join(abs_mode_dir, name_dir)
            for i, img_path in enumerate(os.listdir(abs_name_dir)):
                abs_img_path = os.path.join(abs_name_dir, img_path)
                img = cv2.cvtColor(cv2.imread(abs_img_path), cv2.COLOR_BGR2RGB)

                flipped_img = cv2.flip(img, 1)  # 좌우 반전 이미지
                new_pathf = abs_name_dir + '/' + \
                    name_dir + str(i + 1) + '_F.jpg'
                if(isValidFace(flipped_img)):
                    cv2.imwrite(new_pathf, cv2.cvtColor(
                        flipped_img, cv2.COLOR_RGB2BGR))
                else:
                    print(new_pathf + 'is not Face!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                for angle in angle_list:
                    # 사전 정의된 각도 대로 회전 가함
                    M1 = cv2.getRotationMatrix2D(
                        (img.shape[1]/2, img.shape[0]/2), angle, 1)
                    rotated_img = cv2.warpAffine(
                        img, M1, (img.shape[1], img.shape[0]))
                    M2 = cv2.getRotationMatrix2D(
                        (flipped_img.shape[1]/2, flipped_img.shape[0]/2), angle, 1)
                    flipped_rotated_img = cv2.warpAffine(
                        flipped_img, M2, (flipped_img.shape[1], flipped_img.shape[0]))

                    new_path1 = abs_name_dir + '/' + \
                        name_dir + str(i + 1) + '_' + str(angle) + 'R.jpg'
                    new_path2 = abs_name_dir + '/' + \
                        name_dir + str(i + 1) + '_F_' + str(angle) + 'R.jpg'

                    if isValidFace(rotated_img):
                        print(new_path1)
                        cv2.imwrite(new_path1, cv2.cvtColor(
                            rotated_img, cv2.COLOR_RGB2BGR))
                    else:
                        print(new_path1 + 'is not Face!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                    if isValidFace(flipped_rotated_img):
                        print(new_path2)
                        cv2.imwrite(new_path2, cv2.cvtColor(
                            flipped_rotated_img, cv2.COLOR_RGB2BGR))
                    else:
                        print(new_path2 +
                              '  is not Face!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('==================================================')


if __name__ == '__main__':
    # FaceDatasetMaker('photos')
    DataAugmenter('data')
