import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


def vertical_reflect(img):
    newImg = np.copy(img)
    newImg = newImg[::-1,:]
    return newImg

def horizontal_reflect(img):
    newImg = np.copy(img)
    newImg = newImg[:,::-1]
    return newImg

def translate(img, pixels=10):
    newImg = np.zeros(img.shape, dtype=img.dtype)
    newImg[:,pixels:] = img[:,:-pixels]
    return newImg

def rotate(img, degrees=90):
    newImg = np.copy(img)
    h,w,c = newImg.shape
    for i in range(h):
        for k in range(c):
            newImg[:,h-i-1,k] = img[i,:,k]
    if degrees == 180:
        newImg = rotate(newImg)
    elif degrees == 270:
        newImg = rotate(newImg)
        newImg = rotate(newImg)
    return newImg

def remove(img, area=0.1):
    newImg = np.copy(img)
    h,w,c = newImg.shape
    rec_area = area * h * w
    rec_height = rec_width = int(np.sqrt(rec_area))
    i = np.random.randint(0, h-rec_height)
    j = np.random.randint(0, w-rec_width)

    newImg[i:i+rec_height, j:j+rec_width] = 0

    return newImg

    
if __name__ == "__main__":

    lena = mpimg.imread('lena512color.tiff')

    pixels = 100
    
    translated = translate(lena, pixels)
    mpimg.imsave("translated.png", translated)
    vert_reflected = vertical_reflect(lena)
    mpimg.imsave("vert_reflected.png", vert_reflected)
    hori_reflected = horizontal_reflect(lena)
    mpimg.imsave("hori_reflected.png", hori_reflected)
    rotated_90 = rotate(lena)
    mpimg.imsave("rotated_90.png", rotated_90)
    rotated_180 = rotate(lena, 180)
    mpimg.imsave("rotated_180.png", rotated_180)
    rotated_270 = rotate(lena, 270)
    mpimg.imsave("rotated_270.png", rotated_270)
    removed = remove(lena)
    mpimg.imsave("removed.png", removed)

    plt.subplot(241)
    plt.imshow(lena)
    plt.title("Original")

    plt.subplot(242)
    plt.imshow(translated)
    plt.title("Right translate by {}".format(pixels))

    plt.subplot(243)
    plt.imshow(hori_reflected)
    plt.title("Horizontal reflect")

    plt.subplot(244)
    plt.imshow(vert_reflected)
    plt.title("Vertical reflect")

    plt.subplot(245)
    plt.imshow(rotated_90)
    plt.title("Rotated by 90")

    plt.subplot(246)
    plt.imshow(rotated_180)
    plt.title("Rotated by 180")

    plt.subplot(247)
    plt.imshow(rotated_270)
    plt.title("Rotated by 270")
    
    plt.subplot(248)
    plt.imshow(removed)
    plt.title("Removed 10% area")

    plt.show()