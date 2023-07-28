import concurrent.futures
import math
import os
import re
import time
import cv2
import numpy as np
import pandas as pd
import pytesseract
from scipy import ndimage
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor

def Rotation(imag):
    img_before = imag

    img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, np.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    angles = []

    for [[x1, y1, x2, y2]] in lines:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)
    img_rotated = cv2.rotate(img_before, cv2.ROTATE_90_CLOCKWISE)
    if median_angle != 0.0:
        img_rotated = ndimage.rotate(img_before, median_angle)

    print(f"Угол поворота: {median_angle:.04f}")
    return img_rotated


def in_text(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 3)
    img = cv2.bilateralFilter(img, 3, 35, 35)

    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    img = cv2.merge(result_planes)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=2)
    skel = img.copy()
    kernel = np.ones((3, 3), np.uint8)
    erod = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    temp = cv2.morphologyEx(erod, cv2.MORPH_DILATE, kernel)
    temp = cv2.subtract(img, temp)
    skel = cv2.bitwise_or(skel, temp)
    img2 = cv2.medianBlur(skel, 3)
    img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    result = pytesseract.image_to_string(img2, lang='rus')
    return result


def process_file(file_path):
    imag = convert_from_path(file_path, 300, first_page=1, last_page=1)
    imag[0].save('out.png', 'PNG')
    imag = cv2.imread('out.png', cv2.IMREAD_UNCHANGED)

    img = Rotation(imag)
    res = in_text(img)
    resulat1 = res.split()
    result, df_result = findition(resulat1)

    return df_result


def findition(result, img):
    rekvis = ['инн', 'бик', 'кпп', 'инн.', 'кпп.', 'бик.', 'счета', 'счет', 'счёта', 'счёт', 'место', 'рождения']
    namescp = ['ооо', 'оао', 'наименование', 'общество', 'сумма', 'инн']  # Замените на фактические ключевые слова
    ALLinf = []
    name_org = []
    rekvisiti = []
    df = pd.DataFrame(columns=['Организация', 'Найденные реквизиты'])
    s = 0

    start_time = time.time()

    for i, word in enumerate(result):
        p = word.lower()

        if p in namescp and re.match(r'[«"]', result[i + 1]) and (result[i + 1] + " ") not in ALLinf:
            ALLinf.append(p + " : ")
            name_org.append(p + " : ")

            if rekvisiti:
                df.loc[s, 'Найденные реквизиты'] = rekvisiti
                rekvisiti = []
                s += 1
            else:
                df.loc[s, 'Найденные реквизиты'] = 'Не найдено'
                rekvisiti = []
                s += 1

            k = 1
            while True:
                ALLinf.append(result[i + k] + " ")
                name_org.append(result[i + k] + " ")

                if result[i + k].endswith(("»", "».", "»,", '".', '",', '"')):
                    df.loc[s, 'Организация'] = name_org
                    name_org = []
                    break

                k += 1

        elif p in rekvis and re.match(r'\d', result[i + 1]):
            ALLinf.append(p + " : " + result[i + 1])
            rekvisiti.append(p + " : " + result[i + 1])

            # Добавляем текст на изображение
            text = p + " : " + result[i + 1]
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    print("---%s seconds ---Работа с текстом" % (time.time() - start_time))
    print(result)

    return result, df


def process_file(file_path):
    imag = convert_from_path(file_path, 300, first_page=1, last_page=1)
    imag[0].save('out.png', 'PNG')
    img = cv2.imread('out.png', cv2.IMREAD_UNCHANGED)

    img = Rotation(img)
    res = in_text(img)
    resulat1 = res.split()
    result, df_result = findition(resulat1, img)

    # Сохраняем изображение с текстом
    cv2.imwrite('out_with_text.png', img)

    return df_result


if __name__ == "__main__":
    init_path = r"C:\Users\user\PycharmProjects\Автоматизация документооборота"
    path = os.path.join(init_path, '123')
    files = os.listdir(path)
    df = pd.DataFrame(columns=['Организация', 'Найденные реквизиты'])
    s = 0

    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, os.path.join(path, file)): file for file in files}
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                df_result = future.result()
                df = pd.concat([df, df_result], ignore_index=True)
            except Exception as exc:
                print(f"Error processing {file}: {exc}")

    df.to_excel('primer.xlsx')