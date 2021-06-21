from PIL import Image
from tqdm.autonotebook import tqdm
import numpy as np
import torch
import cv2
import os

from arch import Model
from pipeline import get_transform_mat, get_image_hull_mask, Preprocess

def blursharpen (img, sharpen_mode=0, kernel_size=3, amount=100):
    if kernel_size % 2 == 0:
        kernel_size += 1
    if amount > 0:
        if sharpen_mode == 1: #box
            kernel = np.zeros( (kernel_size, kernel_size), dtype=np.float32)
            kernel[ kernel_size//2, kernel_size//2] = 1.0
            box_filter = np.ones( (kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
            kernel = kernel + (kernel - box_filter) * amount
            return cv2.filter2D(img, -1, kernel)
        elif sharpen_mode == 2: #gaussian
            blur = cv2.GaussianBlur(img, (kernel_size, kernel_size) , 0)
            img = cv2.addWeighted(img, 1.0 + (0.5 * amount), blur, -(0.5 * amount), 0)
            return img
    elif amount < 0:
        n = -amount
        while n > 0:

            img_blur = cv2.medianBlur(img, 5)
            if int(n / 10) != 0:
                img = img_blur
            else:
                pass_power = (n % 10) / 10.0
                img = img*(1.0-pass_power)+img_blur*pass_power
            n = max(n-10,0)

        return img
    return img


def merge_image(model, im):

    # inefficient way of getting face_landmarks
    
    Image.fromarray((im*255).astype(np.uint8)).save("test.jpg")

    pr = Preprocess(model.res)
    pr.do_pipeline("test.jpg")
    img_face_landmarks = pr.face_landmarks

    input_size = model.res
    mask_subres_size = input_size*4
    output_size = input_size

    face_mat = get_transform_mat(img_face_landmarks, output_size)
    face_output_mat = face_mat
    face_mask_output_mat = get_transform_mat(img_face_landmarks, mask_subres_size)

    img_size = im.shape[1], im.shape[0]
    img_face_mask_a = get_image_hull_mask(im.shape, img_face_landmarks)

    dst_face_bgr = cv2.warpAffine(im, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC)
    dst_face_bgr = np.clip(dst_face_bgr, 0, 1)

    dst_face_mask_a_0 = cv2.warpAffine(img_face_mask_a, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC)
    dst_face_mask_a_0 = np.clip(dst_face_mask_a_0, 0, 1)

    model_input = torch.tensor(cv2.resize(dst_face_bgr, (input_size, input_size)), device="cuda")
    pred_src_dst, pred_src_dstm = model(model_input.permute(2, 0, 1)[None, ...], "SRC")

    pred_face_bgr = np.clip(pred_src_dst[0].permute(1, 2, 0).detach().cpu().numpy(), 0, 1)
    pred_face_std_mask_a_0 = np.clip(pred_src_dstm[0].detach().cpu().numpy().reshape((input_size, input_size)), 0, 1)

    img_face_seamless_mask_a = None
    for i in range(1, 10):
        a = img_face_mask_a > i / 10.0
        if len(np.argwhere(a)) == 0:
            continue
        img_face_seamless_mask_a = img_face_mask_a.copy()
        img_face_seamless_mask_a[a] = 1.0
        img_face_seamless_mask_a[img_face_seamless_mask_a <= i / 10.0] - 0.0
        break

    assert img_face_seamless_mask_a is not None

    out_img = cv2.warpAffine(pred_face_bgr, face_output_mat, img_size, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
    out_img = np.clip(out_img, 0.0, 1.0)

    for _ in range(1):
        out_img = blursharpen(out_img, 2, 3, 10)
        out_img = cv2.medianBlur(out_img, 5)
        # out_img = cv2.medianBlur(out_img, 5)

    l,t,w,h = cv2.boundingRect((img_face_seamless_mask_a*255).astype(np.uint8))
    s_maskx, s_masky = int(l+w/2), int(t+h/2)
    out_img = cv2.seamlessClone((out_img*255).astype(np.uint8), (im*255).astype(np.uint8), 
                                (img_face_seamless_mask_a*255).astype(np.uint8), (s_maskx,s_masky) , cv2.NORMAL_CLONE)
    out_img = out_img.astype(dtype=np.float32) / 255.0

    os.remove("test.jpg")

    return out_img

def merge_video(model, in_video_path, out_video_path):
    vidcap = cv2.VideoCapture(in_video_path)
    n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    video = None
    for _ in tqdm(range(n_frames)):
        success, image = vidcap.read()
        if not success:
            break
        
        out_im = merge_image(model, (image/255.0).astype(np.float32))[..., [2, 1, 0]]

        if video is None:
            height, width, channels = out_im.shape
            video = cv2.VideoWriter(out_video_path, 0, 30, (width, height))

        video.write((out_im*255).astype(np.uint8))
        
        image = image[..., [2, 1, 0]]

    cv2.destroyAllWindows()
    video.release()
