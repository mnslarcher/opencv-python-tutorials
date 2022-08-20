###############
# Simple Demo #
###############

import cv2 as cv
import numpy as np

events = [i for i in dir(cv) if i.startswith("EVENT_")]
print(events)


def draw_circle(event, x, y, flags, param):
    if (
        event == cv.EVENT_LBUTTONDBLCLK
    ):  # indicates that left mouse button is double clicked
        cv.circle(img, (x, y), 100, (255, 0, 0), -1)


# Create a black image, a window and bind the function to window
img = np.zeros((512, 512, 3), np.uint8)
cv.namedWindow("image")
cv.setMouseCallback("image", draw_circle)  # window name, mouse callback

while True:
    cv.imshow("image", img)
    k = cv.waitKey(20)
    # 0xFF = 11111111 in binary, with & it cancels everything after the first 8 bits
    # 27 = ord(ESC), can be different with NumLock active, &0xFF solve the problem
    if k & 0xFF == 27:
        break

cv.destroyAllWindows()


######################
# More Advanced Demo #
######################

drawing = False  # True if mouse is pressed
mode = True  # if True draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1

# mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if (
        event
        == cv.EVENT_LBUTTONDOWN  # indicates that the left mouse button is pressed.
    ):
        drawing = True
        ix, iy = x, y
    # EVENT_MOUSEMOVE indicates that the mouse pointer has moved over the window.
    # If this part is removed, nothing is drawn until the left mouse button is released.
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv.circle(img, (x, y), 5, (0, 0, 255), -1)
    elif event == cv.EVENT_LBUTTONUP:  # indicates that left mouse button is released.
        drawing = False
        if mode == True:
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        else:
            cv.circle(img, (x, y), 5, (0, 0, 255), -1)


img = np.zeros((512, 512, 3), np.uint8)
cv.namedWindow("image")
cv.setMouseCallback("image", draw_circle)

while True:
    cv.imshow("image", img)
    k = cv.waitKey(1) & 0xFF
    if k == ord("m"):
        mode = not mode
    elif k == 27:
        break

cv.destroyAllWindows()
