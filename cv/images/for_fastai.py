import cv2
import numpy asa mp
import fastai


def pil2tensor(image, dtype: np.dtype):
    "Borrowed from fast.ai: Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim == 2:
        a = np.expand_dims(a, 2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    return torch.from_numpy(a.astype(dtype, copy=False))


def open_image(fn, div: bool = True, convert_mode: str = 'RGB') -> Image:
    "Borrowed from fast.ai: Return `Image` object created from image in file `fn`."
    x = PIL.Image.open(fn).convert(convert_mode)
    x = pil2tensor(x, np.float32)
    if div:
        x.div_(255)
    return fastai.vision.image.Image(x)


def open_aptos2019_image_v2(fn, convert_mode, after_open) -> Image:
    """
    Create circular crop around image centre
    """
    # TODO check
    sigmaX = 10

    img = fn

    img = cv2.imread(img)
    img = crop_image_from_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    return Image(pil2tensor(img, np.float32).div_(255))
