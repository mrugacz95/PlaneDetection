import cv2
import random
import numpy as np
from skimage import io


def twoDigitsStr(n):
    ret = str(n)
    return '0' + ret if len(ret) <= 1 else ret


def main():
    path = 'http://www.cs.put.poznan.pl/wjaskowski/pub/teaching/kck/labs/planes/samolot'
    for i in range(0, 21):
        im = io.imread(path + twoDigitsStr(i) + '.jpg')
        print('Downloaded: samolot' + twoDigitsStr(i))
        io.imsave('samolot' + str(i) + '.jpg', im)
        name = 'samolot' + twoDigitsStr(i)
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imgray, (5, 5), 3)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
        canny = cv2.Canny(th3, 10, 130, 3)
        kernel = np.ones((6, 6), np.uint8)
        canny = cv2.dilate(canny, kernel, iterations=3)
        im2, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        print('contours number: ' + str(len(contours)))
        hierarchy = hierarchy[0]
        for component in zip(contours, hierarchy):
            cnt, hier = component
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 50 or h < 50:
                continue
            if not hier[3] == -1:
                continue
            color = (random.randrange(0, stop=255), random.randrange(0, stop=255), random.randrange(0, stop=255))
            im2 = cv2.polylines(im, cnt, True, color, thickness=3, lineType=cv2.LINE_AA)
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(im, (cx, cy), radius=5, color=(255, 255, 255), thickness=-1)
            cv2.imwrite(name + 'READY' + '.jpg', im2)
            print('Saved samolot'+twoDigitsStr(i))


if __name__ == '__main__':
    main()
