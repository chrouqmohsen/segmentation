import cv2
import numpy as np

# Step 1: Show image and enable user to select ROI
image = cv2.imread("image.png") 
roi = cv2.selectROI("Select ROI", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 2: Calculate average R,G,B of ROI
x, y, w, h = roi
selected_roi = image[y:y+h, x:x+w]
average_color = np.mean(selected_roi, axis=(0, 1))

# Step 3: Loop over image pixels, calculate distance, and compare with threshold T
T = 50  # Threshold value
binary_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        pixel = image[i, j]
        distance = np.linalg.norm(pixel - average_color)
        if distance < T and pixel[2] > pixel[1] and pixel[2] > pixel[0]:
            binary_image[i, j] = 255  # White pixel (player with red t-shirt)

# Step 4: Apply any morphological operations
kernel = np.ones((5, 5), np.uint8)
binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# Show binary image
cv2.imshow("Binary Image", binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()