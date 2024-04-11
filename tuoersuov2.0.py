import cv2
import pytesseract
import numpy as np
# import threading
import re


def split_string_by_length(input_string, length):
    return [input_string[i:i+length] for i in range(0, len(input_string), length)]
def reverse_string(input_string):
    return input_string[::-1]

# 定义处理每个轮廓的函数
def process_contour(contour):
    x, y, w, h = cv2.boundingRect(contour)
    if w > 20:
        character_image = gray[y:y+h, x:x+w]
        custom_config = r'--psm 6 digits'  # 自定义配置参数
        text = pytesseract.image_to_string(character_image, lang='chi_sim', config=custom_config)
        print("识别结果：", text)
        
        
# 读取图像
image = cv2.imread("3.png")

# 将图像转换为灰度
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("1",gray)
# 对图像进行预处理（如去噪、二值化等）
ret,binary = cv2.threshold(gray,210,255,cv2.THRESH_BINARY_INV)
# _, thresholded_image = cv2.threshold(binary, 200, 255, cv2.THRESH_BINARY)
# cv2.imshow("0",binary)

height, width, _ = image.shape
bin1 = cv2.resize(binary,(width,height))
# cv2.imshow("1",bin1)
# 使用Tesseract OCR进行识别
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# cv2.imshow('1',binary)
# contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# for contour in contours:
#     # 获取轮廓的边界框
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.drawContours(binary, [contour], 0, (0, 0, 255), 2)
#     # 裁剪出每个字母/数字的图像块
#     character_image = gray[y:y+h, x:x+w]
contour_image = np.zeros_like(bin1)
# cv2.imshow('1',contour_image)
contours2=[cnt for cnt in contours ]#过滤太小的contour
# contours3=[cnt for cnt in contours2 if cv2.contourArea(cnt)<8000]#过滤太da的contour
cv2.drawContours(contour_image, contours2, -1, (255,255,255), thickness=cv2.FILLED)

contour_image = cv2.bitwise_not(contour_image)
# cv2.imshow('2',contour_image)
contours, hierarchy = cv2.findContours(contour_image,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
length=0

# # 创建两个线程来处理轮廓
# thread1 = threading.Thread(target=process_contour, args=(contours[0],))
# thread2 = threading.Thread(target=process_contour, args=(contours[1],))

# # 启动线程
# thread1.start()
# thread2.start()

# # 等待两个线程完成
# thread1.join()
# thread2.join()
# 初始化一个空白图像块用于拼接
concatenated_image = None
max_height = 0
for contour in contours:
    # 获取轮廓的边界框
    x, y, w, h = cv2.boundingRect(contour)
    
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 裁剪出每个字母/数字的图像块
    if w > 20:
        length+=1
        character_image = gray[y+5:y+h, x:x+w-5]
        
        # 调整图像大小使其高度相同
        if character_image.shape[0] > max_height:
            max_height = character_image.shape[0]
        
        character_image_resized = cv2.resize(character_image, (character_image.shape[1], max_height))
        
        if concatenated_image is None:
            concatenated_image = character_image_resized
        else:
            concatenated_image = np.hstack((concatenated_image, character_image_resized))
    # if(w>20):
    #     cv2.drawContours(contour_image, [contour], 0, (0, 0, 255), 1)
    #     character_image = gray[y:y+h, x:x+w]
    #     length+=1
        # print(w,h)
       
    # cv2.waitKey(0)
print(length)
# cv2.imshow('2',concatenated_image)
custom_config = r'--psm 6 digits'  # 自定义配置参数
text = pytesseract.image_to_string(concatenated_image,lang='chi_sim',config=custom_config)#,lang='chi_sim',config='-psm 9'
# 打印识别结果
print("识别结果",text)
# text_without_spaces = text.replace(" ", "")
# 使用正则表达式去除空格和换行符
text_without_spaces_and_newlines = re.sub(r'\s+', '', text)
print(len(text_without_spaces_and_newlines))
# print(text_without_spaces_and_newlines+"xx")
# 将字符串分割成每行输出10个字符
lines = split_string_by_length(text_without_spaces_and_newlines, 10)

# # 输出每行的内容
# for line in lines:
#     print(line)
   
# 反转子字符串列表
lines.reverse()

# 逆序输出每行的内容
for line in (lines):
    print(reverse_string(line))     



cv2.waitKey(0)
