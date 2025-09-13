import matplotlib.pyplot as plt
import cv2
import numpy as np


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def ROI(image):
    height = image.shape[0]
    triangle = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_line(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * (3 / 5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])
    except Exception as e:
        return None  # return None if unpacking fails

def average_slope(image, lines):
    left_fit = []
    right_fit = []

    if lines is None:
        return []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 - x1 == 0:
            continue  # avoid division by zero
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    averaged_lines = []

    if left_fit:
        left_avg = np.average(left_fit, axis=0)
        left_line = coordinates(image, left_avg)
        if left_line is not None:
            averaged_lines.append(left_line)

    if right_fit:
        right_avg = np.average(right_fit, axis=0)
        right_line = coordinates(image, right_avg)
        if right_line is not None:
            averaged_lines.append(right_line)

    return np.array(averaged_lines)

# Run on video
cap = cv2.VideoCapture("test2.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    canny_image = canny(frame)
    cropped_image = ROI(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), 40, 5)
    averaged_lines = average_slope(frame, lines)
    line_image = display_line(frame, averaged_lines)
    final_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    cv2.imshow("Result", final_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
