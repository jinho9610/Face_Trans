from mtcnn import MTCNN
from PIL import Image
import cv2
import math
import model_prediction

detector = MTCNN()

model = model_prediction.model
categories = ["hyeontae", "jaehyeon", "jinho", "joohyeong", "yoosung"]

if __name__ == '__main__':
    name = input("name: ")
    cap = cv2.VideoCapture('videos_for_dataset/' + name + '.mp4')

    cnt, frame_cnt = 1, 1
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 프레임에서 얼굴을 탐색한다.
        # MTCNN은 PIL 기반이므로 채널 정보 변경
        infos = detector.detect_faces(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))[0]

        #######################눈을 기준으로 전체적인 frame 수평을 맞춘다#######################
        left_eye_x = infos['keypoints']['left_eye'][0]  # 왼쪽 눈 x
        left_eye_y = infos['keypoints']['left_eye'][1]  # 왼쪽 눈 y
        right_eye_x = infos['keypoints']['right_eye'][0]  # 오른쪽 눈 x
        right_eye_y = infos['keypoints']['right_eye'][1]  # 오른쪽 눈 y

        # 눈 수평을 맞추기 위한 눈 각도 계산
        theta = math.degrees(
            math.atan(-(left_eye_y-right_eye_y) / (right_eye_x - left_eye_x)))
        h, w, c = frame.shape
        matrix = cv2.getRotationMatrix2D((w/2, h/2), theta, 1)
        frame = cv2.warpAffine(frame, matrix, (w, h))
        #################################수평 맞추기 종료###################################

        ##################### 얼굴 중심을 기준으로 정방형으로 검출하고 정방형 resize ####################
        # 수평 조절한 이후 얼굴 재검출
        infos = detector.detect_faces(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))[0]
        box = infos['box']
        face_center_x = box[0] + int(1/2 * box[2])
        face_center_y = box[1] + int(1/2 * box[3])
        a = int(1/2 * box[2]) if box[2] > box[3] else int(1/2 * box[3])
        # 최대한 얼굴 형태 유지하며 정방형으로 잘라냄
        face = frame[face_center_y -
                     a:face_center_y + a, face_center_x - a:face_center_x + a]
        # 얼굴 (100, 100)로 만듦
        face = cv2.resize(face, dsize=(100, 100))
        ##################################### 정방형 resize 종료 ##################################

        print(frame_cnt, model_prediction.predict_by_model(Image.fromarray(
            cv2.cvtColor(face, cv2.COLOR_BGR2RGB)), categories))

        # if frame_cnt % 15 == 0:
        #     cv2.imwrite('videos_for_dataset/' + name +
        #                 '/' + name + '_' + str(cnt) + '.jpg', face)
        #     cnt += 1
        #     if cnt == 29:
        #         break
        frame_cnt += 1
