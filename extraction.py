import cv2
import numpy as np

input_image = "1.png"
output_image = "extracted.png"
img = cv2.imread(input_image)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=1)

contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    pad = 100
    x = max(x - pad, 0)
    y = max(y - pad, 0)
    w = min(w + 2 * pad, img.shape[1] - x)
    h = min(h + 2 * pad, img.shape[0] - y)
    cropped_signature = img[y:y+h, x:x+w]


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_crop = cv2.cvtColor(cropped_signature, cv2.COLOR_BGR2GRAY)
enhanced = clahe.apply(gray_crop)

thresh = cv2.adaptiveThreshold(
    enhanced, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    15, 6
)

kernel = np.ones((2, 2), np.uint8)
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

x_min, y_min = cleaned.shape[1], cleaned.shape[0]
x_max, y_max = 0, 0
for cnt in valid_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    x_min, y_min = min(x_min, x), min(y_min, y)
    x_max, y_max = max(x_max, x + w), max(y_max, y + h)

pad = 20
x_min = max(x_min - pad, 0)
y_min = max(y_min - pad, 0)
x_max = min(x_max + pad, cleaned.shape[1])
y_max = min(y_max + pad, cleaned.shape[0])
cropped_mask = cleaned[y_min:y_max, x_min:x_max]

h, w = cropped_mask.shape
rgba = np.zeros((h, w, 4), dtype=np.uint8)
rgba[:, :, 3] = cropped_mask
rgba[:, :, 0:3] = 0

resized = cv2.resize(rgba, (300, 100), interpolation=cv2.INTER_AREA)

cv2.imwrite(output_image, resized)
