import cv2
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    resized = cv2.resize(opened, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return resized

def preprocess_image_for_llm(image):
    # Nếu ảnh là BGR, chuyển về GRAY
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Resize nhẹ để dễ nhìn hơn
    resized = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # KHÔNG nhị phân hóa, KHÔNG đảo màu
    return resized

def remove_noise(img, pass_factor):
    for column in range(img.size[0]):
        for line in range(img.size[1]):
            value = remove_noise_by_pixel(img, column, line, pass_factor)
            img.putpixel((column, line), value)
    return img

def remove_noise_by_pixel(img, column, line, pass_factor):
    if img.getpixel((column, line)) < pass_factor:
        return 0
    return img.getpixel((column, line))



if __name__ == "__main__":
    image = cv2.imread("test/2xx83.png")
    cv2.imshow("Original", image)
    preprocessed = preprocess_image(image)
    cv2.imshow("Preprocessed", preprocessed)
    preprocessed_for_llm = preprocess_image_for_llm(image)
    cv2.imshow("Preprocessed for LLM", preprocessed_for_llm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()