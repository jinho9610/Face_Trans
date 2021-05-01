'''
torch와 torchvision version info
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
'''

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import mtcnn
import cv2
import time
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import numpy as np

mtcnn = MTCNN(image_size=240, margin=0, keep_all=True,
              min_face_size=40)  # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval()

dest_dir = 'data'
angle_list = [-10, -5, -3, 2, 7, 9]


def collate_fn(x):
    return x[0]


def isValidFace(img):
    img_cropped_list, prob_list = mtcnn(img, return_prob=True)

    if img_cropped_list is not None:
        return True
    else:
        return False


def detect_face(mode):
    if mode == 'train':
        print("making dataset for TRAIN")
        dataset = datasets.ImageFolder('trans_learn_photos/train')

    elif mode == 'val':
        print("making dataset for VAL")
        dataset = datasets.ImageFolder('trans_learn_photos/val')

    # accessing names of peoples from folder names
    idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

    loader = DataLoader(dataset, collate_fn=collate_fn)

    i = 1
    prev_name = ''
    for img, idx in loader:
        cur_name = idx_to_class[idx]

        if cur_name != prev_name:
            i = 1
        else:
            i += 1

        img_cropped_list, prob_list = mtcnn(img, return_prob=True)

        if img_cropped_list is not None:
            boxes, _, faces = mtcnn.detect(
                img, landmarks=True)  # detect는 PIL 상태에서 진행
            box = boxes[0]
            img = np.array(img)  # 이미지 다루기 편하게 opencv로 변환

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img[int(box[1]): int(box[3]), int(
                box[0]): int(box[2])]  # 얼굴 부분만 자르기
            try:
                img = cv2.resize(img, dsize=(160, 160))  # 160x160으로 resize
            except:
                pass

            flipped_img = cv2.flip(img, 1)

            if mode == 'train':
                new_name = 't_' + cur_name

                if isValidFace(img):
                    print(dest_dir + '/train/' +
                          cur_name + '/' + new_name + str(i) + '.jpg')
                    cv2.imwrite(
                        dest_dir + '/train/' + cur_name + '/' + new_name + str(i) + '.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if isValidFace(flipped_img):
                    print(dest_dir + '/train/' +
                          cur_name + '/flipped_' + new_name + str(i) + '.jpg')
                    cv2.imwrite(
                        dest_dir + '/train/' + cur_name + '/flipped_' + new_name + str(i) + '.jpg', cv2.cvtColor(flipped_img, cv2.COLOR_BGR2RGB))

                for k in range(2):
                    for angle in angle_list:
                        if k == 0:
                            M = cv2.getRotationMatrix2D(
                                (img.shape[1]/2, img.shape[0]/2), angle, 1)
                            rotated_img = cv2.warpAffine(
                                img, M, (img.shape[1], img.shape[0]))
                            print(dest_dir + '/train/' +
                                  cur_name + '/rotated_' + str(angle) + new_name + str(i) + '.jpg')

                            if isValidFace(rotated_img):
                                cv2.imwrite(dest_dir + '/train/' +
                                            cur_name + '/rotated_' +
                                            str(angle) + new_name + str(i) + '.jpg', cv2.cvtColor(
                                                rotated_img, cv2.COLOR_BGR2RGB))
                            else:
                                print(isValidFace(rotated_img), dest_dir + '/train/' + cur_name +
                                      '/rotated_' +
                                      str(angle) + new_name + str(i) +
                                      '.jpg', cv2.cvtColor(
                                    rotated_img, cv2.COLOR_BGR2RGB))
                        else:
                            M = cv2.getRotationMatrix2D(
                                (flipped_img.shape[1]/2, flipped_img.shape[0]/2), angle, 1)
                            rotated_img = cv2.warpAffine(
                                flipped_img, M, (flipped_img.shape[1], flipped_img.shape[0]))
                            print(dest_dir + '/train/' +
                                  cur_name + '/rotated_f_' + str(angle) + new_name + str(i) + '.jpg')

                            if isValidFace(rotated_img):
                                cv2.imwrite(dest_dir + '/train/' +
                                            cur_name + '/rotated_f_' +
                                            str(angle) + new_name + str(i) + '.jpg', cv2.cvtColor(
                                                rotated_img, cv2.COLOR_BGR2RGB))
                            else:
                                print(isValidFace(rotated_img), dest_dir + '/train/' + cur_name +
                                      '/rotated_f_' +
                                      str(angle) + new_name + str(i) +
                                      '.jpg', cv2.cvtColor(
                                    rotated_img, cv2.COLOR_BGR2RGB))

            elif mode == 'val':

                new_name = 'v_' + cur_name

                print(dest_dir + '/val/' +
                      cur_name + '/' + new_name + str(i) + '.jpg')
                cv2.imwrite(
                    dest_dir + '/val/' + cur_name + '/' + new_name + str(i) + '.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                print(dest_dir + '/val/' +
                      cur_name + '/flipped_' + new_name + str(i) + '.jpg')
                cv2.imwrite(
                    dest_dir + '/val/' + cur_name + '/flipped_' + new_name + str(i) + '.jpg', cv2.cvtColor(flipped_img, cv2.COLOR_BGR2RGB))

                for k in range(2):
                    for angle in angle_list:
                        if k == 0:
                            M = cv2.getRotationMatrix2D(
                                (img.shape[1]/2, img.shape[0]/2), angle, 1)
                            rotated_img = cv2.warpAffine(
                                img, M, (img.shape[1], img.shape[0]))
                            print(dest_dir + '/val/' +
                                  cur_name + '/rotated_' + str(angle) + new_name + str(i) + '.jpg')

                            if isValidFace(rotated_img):
                                cv2.imwrite(dest_dir + '/val/' +
                                            cur_name + '/rotated_' +
                                            str(angle) + new_name + str(i) + '.jpg', cv2.cvtColor(
                                                rotated_img, cv2.COLOR_BGR2RGB))
                            else:
                                print(isValidFace(rotated_img), dest_dir + '/val/' + cur_name +
                                      '/rotated_' +
                                      str(angle) + new_name + str(i) +
                                      '.jpg', cv2.cvtColor(
                                    rotated_img, cv2.COLOR_BGR2RGB))
                        else:
                            M = cv2.getRotationMatrix2D(
                                (flipped_img.shape[1]/2, flipped_img.shape[0]/2), angle, 1)
                            rotated_img = cv2.warpAffine(
                                flipped_img, M, (flipped_img.shape[1], flipped_img.shape[0]))
                            print(dest_dir + '/val/' +
                                  cur_name + '/rotated_f_' + str(angle) + new_name + str(i) + '.jpg')

                            if isValidFace(rotated_img):
                                cv2.imwrite(dest_dir + '/val/' +
                                            cur_name + '/rotated_f_' +
                                            str(angle) + new_name + str(i) + '.jpg', cv2.cvtColor(
                                                rotated_img, cv2.COLOR_BGR2RGB))
                            else:
                                print(isValidFace(rotated_img), dest_dir + '/val/' + cur_name +
                                      '/rotated_f_' +
                                      str(angle) + new_name + str(i) +
                                      '.jpg', cv2.cvtColor(
                                    rotated_img, cv2.COLOR_BGR2RGB))

        prev_name = cur_name


if __name__ == "__main__":
    # detect_face('train')
    detect_face('val')
