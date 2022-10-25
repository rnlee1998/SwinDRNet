import OpenEXR
import imageio
import numpy as np
import cv2

# image = imageio.imread('/home/jun7.shi/2022/robot/datasets/DREDS/test/0000_simDepthImage.exr','exr')
# img = np.array(image)
#
# imageio.imwrite('/home/jun7.shi/2022/robot/datasets/DREDS/ttt.exr', image)

image = cv2.imread('/home/jun7.shi/2022/robot/datasets/DREDS/test/0000_simDepthImage.exr', cv2.IMREAD_UNCHANGED)
cv2.imwrite('/home/jun7.shi/2022/robot/datasets/DREDS/t.exr', image)


img = cv2.imread('/home/jun7.shi/2022/robot/datasets/DREDS/depth_1.png', cv2.IMREAD_UNCHANGED)
img = img/1000
img[img>2] = 0.001
img2 = cv2.resize(img, (640, 360)).astype(np.float32)
cv2.imwrite('/home/jun7.shi/2022/robot/datasets/DREDS/tt.exr', img2)