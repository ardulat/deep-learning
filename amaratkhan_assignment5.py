import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve2d


def conv2d(img, kernel, stride):
    imgSize = img.shape[0]
    filterSize = kernel.shape[0]
    newImg = np.zeros((imgSize-filterSize+1,imgSize-filterSize+1))
    for i in range(imgSize-filterSize):
        for j in range(imgSize-filterSize):
            center_pixel = (img[i:i+filterSize, j:j+filterSize]*kernel).sum()
            newImg[i,j] = center_pixel
    return newImg

def zeropad(img, padsize):

    imgsize = img.shape[0]
    padded_image = np.zeros((imgsize+2*padsize, imgsize+2*padsize))
    padded_image[padsize:imgsize+padsize, padsize:imgsize+padsize] = np.copy(img)

    return padded_image

if __name__ == "__main__":
    
    data = np.genfromtxt('in.txt', delimiter=',')
    
    # parameters:
    # N - input image size
    # F - filter size
    # P - padding
    # S - stride
    N = data.shape[0]
    F = 3
    P = 1
    S = 1

    kernel1 = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    kernel2 = np.array([[1,0,-1],
                        [2,0,-2],
                        [1,0,-1]])

    if ((N+2*P-F)/S+1).is_integer():
        data = zeropad(data, P)
        img1 = conv2d(data, kernel1, S)
        img2 = conv2d(data, kernel2, S)
    else:
        print('Error: you can not do convolutions with given parameters (kernel, padding, stride).')

    plt.subplot(121)
    plt.imshow(img1, cmap='gray')
    plt.title("Sobel horizontal")
    plt.subplot(122)
    plt.imshow(img2, cmap='gray')
    plt.title("Sobel vertical")
    plt.show()