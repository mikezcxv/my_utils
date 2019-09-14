import numpy as np

import PIL
from PIL import ImageOps, ImageEnhance
import fastai.vision
import torch


from .squircle.squircle import *


def if_prob(res1, res2, prob):
    return res1 if np.random.rand() < prob else res2


class Prob(float):
    def __init__(self, value, *args, **kwargs):
        if value > 1 or value < 0:
            raise ValueError(f'Probability value sholud be in a range of 0..1. {value} given')
        float.__init__(value, *args, **kwargs)


def crop_np(l, crop):
    return l[crop[0]:l.shape[0] - crop[0]:1, crop[1]:l.shape[1] - crop[1]:1]


def shift_image(img: np.array, p_x: Prob = 1, p_y: Prob = 1) -> np.array:
    '''
    :param img:
    :param p_x:
    :param p_y:
    :return:
    '''
    shift_x = int(if_prob(np.random.rand(), 0, p_x) * img.shape[0])
    shift_y = int(if_prob(np.random.rand(), 0, p_y) * img.shape[1])
    c = np.concatenate([img[:, shift_x:, :], img[:, :shift_x, :]], axis=1)
    return np.concatenate([c[shift_y:, :, :], c[:shift_y, :, :]], axis=0)


def symmetric_jitter_img(img: np.array, p: float = 1., p2: float = .7,
                         max_cut_percent: float = .1) -> np.array:
    width, height = img.size  # Get dimensions
    rnd = np.random.rand()
    if width > 0 and height > 0 and rnd < p:
        left, right, top, bottom = 0, width, 0, height
        if (rnd * 1e6) % 100 < p2 * 100:
            d = int(max_cut_percent * height * np.random.rand())
            top = d
            #         if (rnd * 1e8) % 100 < p2 * 100:
            bottom = height - d
            if (rnd * 1e2) % 100 < p2 * 100:
                #                 dw = int(max_cut_percent * width * np.random.rand())
                left = d
                right = width - d
                #         if (rnd * 1e4) % 100 < p2 * 100:

        img = img.crop((left, top, right, bottom))
    return img


def gaussian_img_rotation(img: np.array, max_deg: float = 90, p: float = 1):
    if np.random.rand() < p:
        angle = int(np.random.normal(0, max_deg))
        img = img.rotate(angle)
    return img


def cut_img(img, res_ratio=1.25):
    width, height = img.size  # Get dimensions
    if width > 0 and height > 0:
        new_height = int(height / res_ratio)
        top, bottom = (height - new_height) / 2, (height + new_height) / 2
        left, right = 0, width

        # Crop the center of the image
        img = img.crop((left, top, right, bottom))
    return img


def make_brighter_img(img, brightness):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(brightness)


# TODO check
def make_square_img(img, max_size=(224, 224)):
    w, h = img.width, img.height
    img.thumbnail(max_size)

    new_size = max(w, h)
    delta_w, delta_h = new_size - w, new_size - h
    new_im = PIL.ImageOps.expand(img, (delta_w // 2, delta_h // 2, delta_w, delta_h - (delta_h // 2)))
    return new_im


# Kaggles' solution
def crop_image_from_gray(img, tol=7):
    if img.ndim ==2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            channels = []
            for i in range(3):
                channels.append(img[:, :, i][np.ix_(mask.any(1), mask.any(0))])
            img = np.stack([channels[0], channels[1], channels[2]],axis=-1)
        return img


# https://github.com/fastai/fastai/issues/1653
# def _rot90_affine(k:partial(uniform_int, 0, 3)):
#     "Randomly rotate `x` image based on `k` as in np.rot90"
#     if k%2 == 0:
#         x = -1. if k&2 else 1.
#         y = -1. if k&2 else 1.

#         return [[x, 0, 0.],
#                 [0, y, 0],
#                 [0, 0, 1.]]
#     else:
#         x = 1. if k&2 else -1.
#         y = -1. if k&2 else 1.

#         return [[0, x, 0.],
#                 [y, 0, 0],
#                 [0, 0, 1.]]

# rot90_affine = TfmAffine(_rot90_affine)
# tfms = [rot90_affine()]
# get_ex().apply_tfms(tfms, size=224)


def stretch_img(img, max_size=(224, 224), max_deg=10, method='elliptical'):
    img = make_square_img(img, max_size)
    img = gaussian_img_rotation(img, max_deg=max_deg)
    # ['fgs', 'stretch', 'elliptical']
    return to_square(np.array(img), method=method)


def convert_3ch(img: np.ndarray, tol, inertia_px, padding_px, last_n_rows_smoothing):
    R, G, B = 0, 1, 2
    h, w = img[:, :, 0].shape
    for y in range(h - 1):
        left_bound, accumulate_x, cut_left, accumulate_y, accumulate_cut_left = 0, [], 0, [], []
        start = np.argmax((img[y, :, R] > tol) | (img[y, :, G] > tol) | (img[y, :, B] > tol)) - 1
        prev_zero = 0
        for x in range(start, w - 1, 2):
            #             print(y, x, len(accumulate_x))
            v = img[y, x, :]
            if np.any(v > tol):
                if left_bound == 0:
                    left_bound = x
                accumulate_x.append(v)
                cut_left = x
                prev_zero = 0
            else:
                if prev_zero > 1:
                    accumulate_x = []
                prev_zero += 1

            if len(accumulate_x) > (inertia_px + padding_px) // 2:
                accumulate_cut_left.append(cut_left)
                try:
                    if len(accumulate_cut_left) > 3:
                        cut_left = int((accumulate_cut_left[-1] * 50 \
                                        + accumulate_cut_left[-2] * 4 \
                                        + accumulate_cut_left[-3] * 3 \
                                        + accumulate_cut_left[-4] * 2 \
                                        ) / 59)
                    # + accumulate_cut_left[-5] * 2
                    l = cut_left - inertia_px
                    img[y, 0:l:1, :] = np.flip(img[y, l:2 * l:1, :], axis=0)
                except:
                    #                                     print('Ex')
                    #                                     print(f'From 0 to {cut_left} replace with', accumulate_x)
                    accumulate_y.append(accumulate_x[padding_px:])
                    img[y, 0:cut_left - inertia_px:1, :] = np.mean(accumulate_y[-last_n_rows_smoothing:])
                # last_n_rows_smoothing = 100

                break
    return img


def padd_corners(img: np.ndarray, tol=7, inertia_px=20, padding_px=25, last_n_rows_smoothing=30, crop_x=3, crop_y=5,
                     crop_y_last=13):
    # Only for 3 channels. Checkage is copy pasted    , MAX_HEIGHT=800
    if img.ndim == 2:
        return img
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image

            #     if img.shape[0] > MAX_HEIGHT:
            #         img = Image.fromarray(img)
            #         w = img.width
            #         h = img.height
            #         img.thumbnail((int(w * MAX_HEIGHT / h), MAX_HEIGHT), Image.ANTIALIAS)
            #         img = np.array(img)

    img = crop_np(img, [crop_y, crop_x])
    img = convert_3ch(img, tol=tol, inertia_px=inertia_px, padding_px=padding_px,
                      last_n_rows_smoothing=last_n_rows_smoothing)
    img = np.flip(img, axis=1)
    img = convert_3ch(img, tol=tol, inertia_px=inertia_px, padding_px=padding_px,
                      last_n_rows_smoothing=last_n_rows_smoothing)
    img = np.flip(img, axis=1)
    #     img = convert_3ch(img, tol = tol, inertia_px = inertia_px, padding_px = padding_px, last_n_rows_smoothing=last_n_rows_smoothing,
    #                       crop_x=crop_x, crop_y=crop_y, crop_y_last=crop_y_last)
    img = crop_np(img, [crop_y_last, 0])

    return img


# From Kaggle
def circle_crop(img, sigmaX=10):
    """
    Create circular crop around image centre
    """

    img = cv2.imread(img)
    img = crop_image_from_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amax((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    #     img = crop_image_from_gray(img)
    #     img = cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    #     img = cv2.resize(img, img_size)
    return img


