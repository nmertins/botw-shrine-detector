import sys
import cv2
import numpy as np

img_rgb = cv2.imread('./../resources/complete_shrine_map.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

SLIDER_WINDOW_NAME = 'HSV Controls'
HUE_SLIDER_NAME = 'Hue'
HUE_OFFSET_SLIDER_NAME = 'Hue Offset'
SATURATION_LOWER_SLIDER_NAME = 'Saturation Lower'
SATURATION_UPPER_SLIDER_NAME = 'Saturation Upper'
BRIGHTNESS_LOWER_SLIDER_NAME = 'Brightness Lower'
BRIGHTNESS_UPPER_SLIDER_NAME = 'Brightness Upper'

HUE_INDEX = 0
HUE_OFFSET_INDEX = 1
SATURATION_LOWER_INDEX = 2
SATURATION_UPPER_INDEX = 3
BRIGHTNESS_LOWER_INDEX = 4
BRIGHTNESS_UPPER_INDEX = 5


"""
Hue:  108
Hue Offset:  20
Saturation Lower Bound:  80
Saturation Upper Bound:  255
Brightness Lower Bound:  128
Brightness Upper Bound:  255
"""


def apply_note_threshold(fretboard: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(fretboard, cv2.COLOR_BGR2HSV)

    np.clip(lower_bound, 0, 255, out=lower_bound)
    np.clip(upper_bound, 0, 255, out=upper_bound)

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    res = cv2.bitwise_and(fretboard, fretboard, mask=mask)

    return res


def nothing(val: int) -> None:
    pass


def setup_slider_controls() -> None:
    cv2.namedWindow(SLIDER_WINDOW_NAME)
    cv2.resizeWindow(SLIDER_WINDOW_NAME, 600, 200)
    cv2.createTrackbar(HUE_SLIDER_NAME, SLIDER_WINDOW_NAME, 0, 255, nothing)
    cv2.createTrackbar(HUE_OFFSET_SLIDER_NAME, SLIDER_WINDOW_NAME, 10, 20, nothing)
    cv2.createTrackbar(SATURATION_LOWER_SLIDER_NAME, SLIDER_WINDOW_NAME, 100, 255, nothing)
    cv2.createTrackbar(SATURATION_UPPER_SLIDER_NAME, SLIDER_WINDOW_NAME, 255, 255, nothing)
    cv2.createTrackbar(BRIGHTNESS_LOWER_SLIDER_NAME, SLIDER_WINDOW_NAME, 100, 255, nothing)
    cv2.createTrackbar(BRIGHTNESS_UPPER_SLIDER_NAME, SLIDER_WINDOW_NAME, 255, 255, nothing)


def read_slider_values() -> tuple:
    hue = cv2.getTrackbarPos(HUE_SLIDER_NAME, SLIDER_WINDOW_NAME)
    hue_offset = cv2.getTrackbarPos(HUE_OFFSET_SLIDER_NAME, SLIDER_WINDOW_NAME)
    saturation_lower = cv2.getTrackbarPos(SATURATION_LOWER_SLIDER_NAME, SLIDER_WINDOW_NAME)
    saturation_upper = cv2.getTrackbarPos(SATURATION_UPPER_SLIDER_NAME, SLIDER_WINDOW_NAME)
    brightness_lower = cv2.getTrackbarPos(BRIGHTNESS_LOWER_SLIDER_NAME, SLIDER_WINDOW_NAME)
    brightness_upper = cv2.getTrackbarPos(BRIGHTNESS_UPPER_SLIDER_NAME, SLIDER_WINDOW_NAME)

    return hue, hue_offset, saturation_lower, saturation_upper, brightness_lower, brightness_upper


def get_bounds() -> tuple:
    slider_values = read_slider_values()
    hue = slider_values[HUE_INDEX]
    hue_offset = slider_values[HUE_OFFSET_INDEX]
    saturation_lower = slider_values[SATURATION_LOWER_INDEX]
    saturation_upper = slider_values[SATURATION_UPPER_INDEX]
    brightness_lower = slider_values[BRIGHTNESS_LOWER_INDEX]
    brightness_upper = slider_values[BRIGHTNESS_UPPER_INDEX]

    lower_bound = np.array([hue - hue_offset,
                            saturation_lower,
                            brightness_lower])
    upper_bound = np.array([hue + hue_offset,
                            saturation_upper,
                            brightness_upper])

    return lower_bound, upper_bound


def print_variable_values() -> None:
    slider_values = read_slider_values()
    hue = slider_values[HUE_INDEX]
    hue_offset = slider_values[HUE_OFFSET_INDEX]
    saturation_lower = slider_values[SATURATION_LOWER_INDEX]
    saturation_upper = slider_values[SATURATION_UPPER_INDEX]
    brightness_lower = slider_values[BRIGHTNESS_LOWER_INDEX]
    brightness_upper = slider_values[BRIGHTNESS_UPPER_INDEX]

    print('Hue: ', hue)
    print('Hue Offset: ', hue_offset)
    print('Saturation Lower Bound: ', saturation_lower)
    print('Saturation Upper Bound: ', saturation_upper)
    print('Brightness Lower Bound: ', brightness_lower)
    print('Brightness Upper Bound: ', brightness_upper)


def hsv_tuning() -> int:
    setup_slider_controls()

    while True:
        lower_bound, upper_bound = get_bounds()
        notes = apply_note_threshold(img_rgb, lower_bound, upper_bound)

        cv2.imshow('HSV View', notes)
        cv2.imshow('Shrine Map', img_rgb)

        wait_key = cv2.waitKey(33) & 0xFF

        if wait_key == ord('p'):
            print_variable_values()

        if wait_key == ord(' '):
            grab_new_frame = not grab_new_frame

        if wait_key == ord('q'):
            break

    cv2.destroyAllWindows()

    return 0


def template_matching():
    template = cv2.imread('./../resources/shrine.jpg', 0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.6
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite('./../resources/res.jpg', img_rgb)

    return 0


if __name__ == "__main__":
    sys.exit(hsv_tuning())
    # sys.exit(template_matching())
