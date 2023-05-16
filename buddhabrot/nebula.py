import cv2

red = cv2.imread("red.png")[..., 2]
green = cv2.imread("green.png")[..., 2]
blue = cv2.imread("blue.png")[..., 2]

out = cv2.merge((blue, green, red))
cv2.imwrite("nebula.png", out)
