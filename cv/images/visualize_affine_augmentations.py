import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def find_rotation_aug(is_horizontal=True, is_vertical=True, angle=45, n=500, p=(.5, .5, .5)):
    p1, p2, p3 = p
    w, h = 448, 448
    im = Image.new('RGBA', (w, h), (255, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    # dr.ellipse((0, 0, 112, 112), fill="black")
    # dr.line([(0, h // 2), (w//2, h//2)], fill='yellow', width=2)
    # dr.line([(0, h//2), (w, h//2)], fill='yellow', width=1)

    # dr.line([(w - 3, h//2 - 5), (w, h//2 + 5)], fill='red', width=1)
    # dr.line([(w - 3, h//2 + 5), (w, h//2 - 5)], fill='red', width=1)

    dr.line([(w - 3, h // 2 - 5), (w, h // 2 + 5)], fill='red', width=1)
    dr.line([(w - 3, h // 2 + 5), (w, h // 2 - 5)], fill='red', width=1)

    dr.line([(13, h // 2 - 5), (16, h // 2 + 5)], fill='blue', width=1)
    dr.line([(13, h // 2 + 5), (16, h // 2 - 5)], fill='blue', width=1)

    dr.line([(w // 2 - 3, 23), (w // 2, 28)], fill='green', width=1)
    dr.line([(w // 2 + 3, 26), (w // 2, 31)], fill='green', width=1)

    plt.figure(figsize=(7, 7))

    for i in range(n):
        img = im.copy()
        if np.random.rand() < p1 and angle:
            img = img.rotate(random.uniform(-angle, angle))
        if np.random.rand() < p2 and is_horizontal:
            img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)  # Flip horizontal
        if np.random.rand() < p3 and is_vertical:
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)  # Flip vertical

        plt.imshow(img)
    plt.show()
