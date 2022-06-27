import cv2
import imutils
import numpy as np
#import image_slicer

#image_slicer.slice('C:\\files\\cars.png', 64)


def work():
    # read image
    img = cv2.imread('BR.png')
    imgResized = cv2.resize(img, (960, 540))
    #tiles = [im[x:x + 2857.125, y:y + 3980] for x in range(0, im.shape[0], M) for y in range(0, im.shape[1], N)]
    # cv2_imshow(imgResized)
    #image_slicer.slice('BR.png', 200)

    # channel splitting
    blue, green, red = cv2.split(img)
    # cv2_imshow(green)

    # binarize green component w/ thresholding
    retval, dst = cv2.threshold(green, 160, 255, cv2.THRESH_TOZERO)
    cv2.imshow("dst", dst)

    # retval, dst	=	cv2.threshold(dst, 100, 255, cv2.THRESH_TOZERO_INV)
    # cv2_imshow(dst)
    # contouring

    contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(img.shape[:2], dtype=img.dtype)

    # draw all contours larger than 1000 on the mask
    for c in contours:
        if cv2.contourArea(c) > 60:
            x, y, w, h = cv2.boundingRect(c)
            cv2.drawContours(mask, [c], 0, (255), -1)
            cv2.drawContours(img, [c], 0, (255), 3)

    #maskResized = cv2.resize(mask, (960, 540))
    # cv2_imshow(maskResized)
    # apply the mask to the original image
    # result = cv2.bitwise_and(img,img, mask= mask)
    # cv2_imshow(result)
    # cv2_imshow(image)

    cv2.imshow("mask", mask)

    #img_out_Resized = cv2.resize(img, (960, 540))
    cv2.imshow("img", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# plt -- comp
'''
  fig1 = plt.figure(1, figsize=(20,20))
  fig1.show()
  ax1 = fig1.add_subplot(121)
  fig2 = plt.figure(2, figsize=(20,20))
  fig2.show()
  ax2 = fig2.add_subplot(122)
  ax1.imshow(imgResized)
  ax2.imshow(maskResized)
'''

# final = cv2.drawContours(img, contours, -1, (0,0,0), 3)
# cv2_imshow(final)


if __name__ == '__main__':
    work()