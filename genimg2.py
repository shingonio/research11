import cv2
from datetime import datetime



src_img = cv2.imread('./imgs/00001.jpg')
back_img = cv2.imread('./imgs/hon41.jpg')

height, width, channel = src_img.shape

dst_img = back_img


print(dst_img[10, 100])

for x in range(height):
    for y in range(width-1):
        b, g, r = dst_img[x,y]
        if 50 <= b <= 149 and 50 <= g <= 147 and 50 <= r <= 139:
            dst_img[x,y] = src_img[x,y]

cv2.imwrite("out41.jpg", dst_img)