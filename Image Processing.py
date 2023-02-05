from PIL import Image
import numpy as np
from abc import ABC, abstractmethod
import time
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb


# THINGS TO DO
# organize code properly
# - fix image class
#       - inhereit from image properly, separate data array
# time colourshifting, improve method
# set up github

class ImageArr:
    # data is array of pixels, organized col, row, pixels
    img = ""
    data = []

    def __init__(self, img):
        # img is a PIL image object
        self.img = img
        self.data = np.array(img, np.uint8)

    def show(self):
        # uses PIL to creat a window displaying an image
        Image.fromarray(self.data).show()

    def copy(self):
        return ImageArr(self.data.copy())

    def fft(self, show=False):
        # add show option
        t1 = time.time()
        if show:

            arr = np.array(self.img.convert('L'), np.uint8)
            arr = np.fft.fft2(arr)

            for cindex, col in enumerate(arr):
                for rindex, row in enumerate(col):
                    arr[cindex][rindex] = np.abs(np.mean(arr[cindex][rindex]))

            max = np.log(np.amax(arr))
            for cindex, col in enumerate(arr):
                for rindex, row in enumerate(col):
                    if arr[cindex][rindex].item(0).real > 0.0000001:
                        arr[cindex][rindex] = np.log(arr[cindex][rindex])/max*255

            arr = arr.astype(np.uint8)
            arr = np.fft.fftshift(arr)
            # Image.fromarray(arr).show()
            # plt.imshow(arr, cmap="hot")
            # plt.show()

        arr = np.fft.fft2(self.data)
        print(time.time()-t1)
        return arr








class ImageOperation(ABC):
    # abstract class for image operations
    @abstractmethod
    def modifyImage(self, arr):
        # perform operation on image
        # if additional information is needed (colour shift vector)
        # concrete class should be instantiated with it
        # should return new ImageArr object, leaving original untouched
        pass


# ----------------------------------







class ColourShift(ImageOperation):
    shift = []

    def __init__(self, shift):
        # shift is 3 x 1 vector that represent amount to shift colours by in image
        # (shift.length == 3) ^ (forall i| 0 <= i < 3: -256 < shift[i] < 256)
        self.shift = shift

    def setShift(self, shift):
        # shift is 3 x 1 vector that represent amount to shift colours by in image
        # (shift.length == 3) ^ (forall i| 0 <= i < 3: -256 < shift[i] < 256)
        self.shift = shift

    def modifyImage(self, img):
        # img -> imagearr instance
        # shift -> list with a shift value for R G B channels
        # (forall c,r | 0 < c < arr.length ^ 0 < r < arr[0].length: arr[c][r].length == 3)
        # (shift.length == 3) ^ (forall i| 0 <= i < 3: -256 < shift[i] < 256)
        # returns new modified copy of arr, leaving original array untouched
        arr2 = img.data.copy()
        for cindex, col in enumerate(arr2):
            for rindex, row in enumerate(col):
                arr2[cindex][rindex] = self.addLimit(arr2[cindex][rindex], self.shift, 255, 0)

        return ImageArr(arr2)

    def addLimit(self, arr, add, maxval, minval):
        # ensures that performing desired addition keeps values within bounds
        # arr -> 1d array of ints
        # add -> id array of ints
        # maxval -> maximum value for an integer in arr
        # minval -> minimum value for an integer in arr
        # (arr.length == add.length) ^ (forall i| i < arr.length: arr[i] >= minval ^ arr[i] <= maxval)
        for index, val in enumerate(arr):
            tval = val + add[index]
            if tval > maxval:
                arr[index] = maxval
            elif tval < minval:
                arr[index] = minval
            else:
                arr[index] = tval
        return arr


# ------------------------------------


def __main__():
    im = Image.open("mcmaster_university.png") # opportunity to encapsulate in imageArr class

    im.thumbnail((128, 129)) #change resolution

    a = ImageArr(np.array(im))

    cs = ColourShift([-251, 7, 119])

    b = a.copy()

    a = cs.modifyImage(b)

    for i in range(25):

        cs.setShift([i*10,i*10,0])
        t1 = time.time()
        c = cs.modifyImage(b)
        print(time.time() - t1)
        c.show()


# __main__()



im = Image.open("mcmaster_university.png")

im2 = Image.open("mcmaster_university.png").convert('L')
# im2.thumbnail((128, 129))

c = ImageArr(im2)
c.fft(True)

# im2 = np.array(im2, np.uint8)
# c = ImageArr(im2)
# c.show()
# c.fft()
#c.show()
#c.data = np.fft.fft2(c.data)
#print(c.data)
#d = c.data.astype(np.uint8)
#Image.fromarray(d).show()
#c.data = np.fft.ifft2(c.data).astype(np.uint8)
#c.show()
# im.thumbnail((128, 128))

