import os
import sys
import logging
import time
import argparse
import numpy as np
import natsort
import torch
import time
import cv2
import torchvision.transforms as transforms
import utils

from collections import OrderedDict
from PIL import Image
from mtcnn import MTCNN
from glob import glob
from os.path import isfile, join

from models.SRGAN_model import SRGANModel
from glob import glob
from utils import check_args

#from jh_FaceDetection import FaceDetector

#from Alignment import detect_face, face_recover,read_cv2_img
_transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                      std=[0.5, 0.5, 0.5])])

# --------------------Option---------------------------------------


def get_FaceSR_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr_G', type=float, default=1e-4)
    parser.add_argument('--weight_decay_G', type=float, default=0)
    parser.add_argument('--beta1_G', type=float, default=0.9)
    parser.add_argument('--beta2_G', type=float, default=0.99)
    parser.add_argument('--lr_D', type=float, default=1e-4)
    parser.add_argument('--weight_decay_D', type=float, default=0)
    parser.add_argument('--beta1_D', type=float, default=0.9)
    parser.add_argument('--beta2_D', type=float, default=0.99)
    parser.add_argument('--lr_scheme', type=str, default='MultiStepLR')
    parser.add_argument('--niter', type=int, default=100000)
    parser.add_argument('--warmup_iter', type=int, default=-1)
    parser.add_argument('--lr_steps', type=list, default=[50000])
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--pixel_criterion', type=str, default='l1')
    parser.add_argument('--pixel_weight', type=float, default=1e-2)
    parser.add_argument('--feature_criterion', type=str, default='l1')
    parser.add_argument('--feature_weight', type=float, default=1)
    parser.add_argument('--gan_type', type=str, default='ragan')
    parser.add_argument('--gan_weight', type=float, default=5e-3)
    parser.add_argument('--D_update_ratio', type=int, default=1)
    parser.add_argument('--D_init_iters', type=int, default=0)

    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--crop_size', type=float, default=0.85)
    parser.add_argument('--lr_size', type=int, default=128)
    parser.add_argument('--hr_size', type=int, default=512)

    # network G
    parser.add_argument('--which_model_G', type=str, default='RRDBNet')
    parser.add_argument('--G_in_nc', type=int, default=3)
    parser.add_argument('--out_nc', type=int, default=3)
    parser.add_argument('--G_nf', type=int, default=64)
    parser.add_argument('--nb', type=int, default=16)

    # network D
    parser.add_argument('--which_model_D', type=str,
                        default='discriminator_vgg_128')
    parser.add_argument('--D_in_nc', type=int, default=3)
    parser.add_argument('--D_nf', type=int, default=32)

    parser.add_argument('--pretrain_model_G', type=str,
                        default='./check_points/ESRGAN-V1/245000_G.pth')
    parser.add_argument('--pretrain_model_D', type=str, default=None)

    # test arguments
    # 기존 비디오 경로
    # parser.add_argument('--ori_video_path', type=str,
    #                     default='/content/drive/MyDrive/Face-Super-Resolution/Video_test/video/test_video5_er.MP4')
    # # 비디오 -> frame에서 frame 경로
    # parser.add_argument('--ori_frame_path', type=str,
    #                     default="Video_test/Test_Frame")
    # # face detection을 위한 frame 경로
    # parser.add_argument('--frame_detection', type=str,
    #                     default='/content/drive/MyDrive/Face-Super-Resolution/Video_test/Test_Frame/*')
    # # face crop 경로
    # parser.add_argument('--crop_path', type=str,
    #                     default='/content/drive/MyDrive/Face-Super-Resolution/Video_test/crop_img')
    # # 개선 후 결과 저장 경로
    # parser.add_argument('--sr_path', type=str, default='Video_test/sr_img/')
    # # final frame path
    # parser.add_argument('--final_path', type=str,
    #                     default="/content/drive/MyDrive/Face-Super-Resolution/Video_test/final")
    # # final video
    # parser.add_argument('--final_video', type=str,
    #                     default='Video_test/final_video/TestFrame.avi')

    args = parser.parse_args()

    return args


def SR(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    print("Start SR test")
    try:
        sr_model = SRGANModel(get_FaceSR_opt(), is_train=False)
    except Exception as e:
        print('no module', e)
    print(1)
    sr_model.load()
    print(2)

    in_img = torch.unsqueeze(_transform(Image.fromarray(img)), 0)
    print(3)
    sr_model.var_L = in_img.to(sr_model.device)
    print(4)
    sr_model.test()
    print(5)
    #visuals = sr_model.fake_H.squeeze(0).cpu().numpy()
    visuals = sr_model.fake_H.detach().float().cpu()
    print(6)
    image_numpy = utils.tensor2im(visuals, show_size=224)
    print(7)
    image_numpy = np.reshape(image_numpy, (-1, 224, 3))
    print(8)
    #image_numpy = cv2.resize(image_numpy, (img.shape[0], img.shape[1]))
    print('End test')
    return image_numpy


# ------------------------------Main---------------------------------------
def main():
    # ---------------------------Test ---------------------------------
    print("Start SR test")
    img = utils.read_cv2_img(i)
    #img = img.resize((128,128))
    in_img = torch.unsqueeze(_transform(Image.fromarray(img)), 0)
    sr_model.var_L = in_img.to(sr_model.device)
    sr_model.test()
    #visuals = sr_model.fake_H.squeeze(0).cpu().numpy()
    visuals = sr_model.fake_H.detach().float().cpu()
    image_numpy = utils.tensor2im(visuals, show_size=317)
    image_numpy = np.reshape(image_numpy, (-1, 317, 3))
    image_numpy = cv2.resize(image_numpy, (img.shape[0], img.shape[1]))
    print('End test')
    print()
    # ----------------------------------End test--------------------------

    # -----------------------------SR img combine Original img --------------------------------------------
    #row,cols, channels = crop_img.shape
    start_x = result[i][j]['box'][0]
    start_y = result[i][j]['box'][1]
    end_x = result[i][j]['box'][0] + result[i][j]['box'][2]
    end_y = result[i][j]['box'][1] + result[i][j]['box'][3]
    area = (start_x, start_y, end_x, end_y)
    crop_img = crop_img.resize((int(end_x - start_x), int(end_y - start_y)))

    px = f_img.load()
    c_px = crop_img.load()
    c_x = 0
    c_x_max = crop_img.width
    c_y = 0
    c_y_max = crop_img.height
    # print(crop_img.width,crop_img.height)
    # print(end_x-start_x,end_y-start_y)
    for q in range(start_x, end_x):
        c_y = 0
        for k in range(start_y, end_y):
            # print(c_px[c_x,c_y])
            try:
                px[q, k] = c_px[c_x, c_y]
                if(c_y < c_y_max):
                    c_y = c_y+1
            except:
                if(c_y < c_y_max):
                    c_y = c_y+1
                pass
        if(c_x < c_x_max):
            c_x = c_x+1
    my_count = my_count+1
    # f_img.save('/content/drive/MyDrive/Face-Super-Resolution/Video_test/final/{}.png'.format(i.split('/')[-1].split('.')[0]))
    f_img.save(args.final_path +
               '/{}.png'.format(i.split('/')[-1].split('.')[0]))
    r_c = r_c+1
    # cv2.imwrite("/content/drive/MyDrive/PARK/IP-FSRGAN/final/{}.png".format(r_c),f_img)
    print("End SR img combine to ori img")
    print()
    # -----------------------------SR img combine Original img End --------------------------------------------
    end_time = time.time()

    print("time = {}".format(end_time - start_time))


# main()
