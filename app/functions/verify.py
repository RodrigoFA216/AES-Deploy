import cv2
import os

def verify_image_color(path, imgFolder, name):
    file_folder = os.path.join(imgFolder[:-4], 'verify', name[:-4])
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(file_folder, img_gray)
    diff_img = cv2.absdiff(file_folder, img_gray)
    sum_diff = diff_img.sum()
    if sum_diff == 0:
        return False
    if sum_diff != 0:
        return True