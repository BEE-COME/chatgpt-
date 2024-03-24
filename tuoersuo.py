import cv2
import pytesseract
import numpy as np
# import threading
import re
import pyautogui
import time
import keyboard
import sys

def stop_program(e):
    if e.name == 'q':
        print("Stopping the program...")
       

X1,Y1,W1,H1=0,0,0,0

def delay(milliseconds):
    time.sleep(milliseconds / 1000)

# # 调用delay函数来实现延迟1秒
# delay(
# delay
# 1000)
# print("延迟1秒后执行这行代码")


def draw_rectangle(x, y, width, height):
    # 移动鼠标到起始点
    pyautogui.moveTo(x, y)

    # 按下鼠标左键
    pyautogui.mouseDown()
    # # 向下拖动鼠标画出矩形的高
    pyautogui.moveRel(width, height, duration=0.01)
    # 松开鼠标左键
    pyautogui.mouseUp()

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
        
# keyboard.on_press(stop_program)
        
        
# 获取屏幕截图
screenshot = pyautogui.screenshot()
# 将截图转换为OpenCV格式
image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
# cv2.imshow("1", image)
# cv2.waitKey(0)
n=5
# image = cv2.imread('98.png')
# resized_image = cv2.resize(image, (1920, 1080))
resized_image = cv2.resize(image, (3840, 2160))
# cv2.imshow("1", resized_image)
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
edges = cv2.Canny(blurred_image, 50, 150)

# dilated_edges = cv2.dilate(edges, None, iterations=1)
# cv2.imshow("1 Contours", dilated_edges)
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 创建一个空白图像
contour_image = np.zeros_like(resized_image)
plate_contour = None
max_area = 0
max_contour = None
# 遍历所有轮廓
for contour in contours:
    # 进行轮廓近似，获取近似的多边形
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 如果近似结果是一个四边形（边数为4），则认为是矩形
    if len(approx) == 4:
        # 绘制矩形轮廓
        cv2.drawContours(contour_image, [approx], 0, (0, 0, 255), 2)
        area = cv2.contourArea(approx)

        # 更新最大面积和对应的轮廓
        if area > max_area:
            max_area = area
            max_contour = approx
    
# cv2.imshow("Rectangular Contours", contour_image)
# cv2.imshow("1 Contours", dilated_edges)
plate_contour=max_contour
# 检查是否找到了车牌区域的轮廓
if plate_contour is not None:
    # 获取车牌区域的边界框
    x, y, w, h = cv2.boundingRect(plate_contour)

    # 从原始图像中提取车牌区域
    plate_image = resized_image[y+n:y+h-n, x+n:x+w-n]
    print(x,y,w,h)
    X1,Y1,W1,H1=x,y,w,h

    # 显示车牌图像
    # cv2.imshow("Plate Image", plate_image)
    cv2.imwrite("3.png",plate_image)

# 显示矩形轮廓图像
# cv2.imshow("Rectangular Contours", contour_image)


        
# 读取图像
image = plate_image
# cv2.imread("3.png")

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


# 初始化一个空白图像块用于拼接
concatenated_image = None
max_height = 0
for contour in contours:
    # 获取轮廓的边界框
    x, y, w, h = cv2.boundingRect(contour)
    
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 裁剪出每个字母/数字的图像块
    # if w > 20:
    if w > 40:
        length+=1
        # character_image = gray[y+5:y+h, x:x+w-5]
        character_image = gray[y+10:y+h-5, x+5:x+w-20]
        
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
# print(length)
# cv2.imshow('2',concatenated_image)
# cv2.waitKey(0)
custom_config = r'--psm 6 digits'  # 自定义配置参数
text = pytesseract.image_to_string(concatenated_image,lang='chi_sim',config=custom_config)#,lang='chi_sim',config='-psm 9'
# 打印识别结果
# print("识别结果",text)
# text_without_spaces = text.replace(" ", "")
# 使用正则表达式去除空格和换行符
text_without_spaces_and_newlines = re.sub(r'\s+', '', text)
# print(len(text_without_spaces_and_newlines))
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
    
# cv2.waitKey(0)
data = np.zeros((16, 10))  # 创建一个5行10列2维的NumPy数组，用于存储X和Y坐标

# 逆序输出每行的内容
row=0
col=0   
for line in (lines):
    for b in reverse_string(line):
        if(col>=10):
            row+=1
            col=0
        data[row, col]=int(b)
        # print(data[row, col])
        col+=1
#     print(reverse_string(line))     
    

for row in range(16):
    for col in range(10):
        print(f"Point ({row}, {col}): data={data[row, col]}")
        # pyautogui.moveTo(coordinates[row, col, 0],coordinates[row, col, 1]) 

delay(1000)
# cv2.waitKey(0)
# 调用函数来画一个矩形，起始点坐标为(100, 100)，宽为300，高为200
print(X1,Y1,W1,H1)
# 矩形70*70
dt_h=(H1-35)/16
dt_w=(W1-30)/10

delay(1000)

add = np.zeros((16, 10, 2))  # 创建一个5行10列2维的NumPy数组，用于存储X和Y坐标

# 设置坐标值
for row in range(16):
    for col in range(10):
        add[row, col, 0] = X1+35+col*dt_w  # 设置X坐标值
        add[row, col, 1] = Y1+30+row*dt_h  # 设置Y坐标值

cnt=0
num=1
# # 打印坐标
# for row in range(16):
#     for col in range(10):
#         # print(f"Point ({row}, {col}): X={add[row, col, 0]}, Y={add[row, col, 1]}")
#         pyautogui.moveTo(add[row, col, 0],add[row, col, 1]) 
#         delay(100)
def process_data_col(data, add):
    tmp = 0
    sum = 0
    found=0
    for row in range(16):
        for col in range(10):
            sum = data[row, col]
            if sum==0:
                continue
            tmp= data[row, col]
            
            for j in range(1,10):
                if sum < 10:
                    if col+j<=9:
                        sum+= data[row, col + j]  
                else:
                    j-=1
                    break              
            
            if sum == 10:
                if cnt<100:
                    if j<num+1:
                        if tmp<1+num or tmp>9-num:
                            draw_rectangle(add[row, col, 0], add[row, col, 1], 50 + add[row, col + j, 0] - add[row, col, 0], 50 + add[row, col + j, 1] - add[row, col, 1])
                            for i in range(j+1):                    
                                data[row, col+i]=0
                            found=1
                            break                    
                else:
                    draw_rectangle(add[row, col, 0], add[row, col, 1], 50 + add[row, col + j, 0] - add[row, col, 0], 50 + add[row, col + j, 1] - add[row, col, 1])
                    for i in range(j+1):                    
                        data[row, col+i]=0
                    found=1
                    break
        if(found):
            break
    return found

def process_data_row(data, add):
    tmp = 0
    sum = 0
    found=0
    for col in range(10):
        for row in range(16):        
            sum = data[row, col]
            if sum==0:
                continue
            tmp= data[row, col]
            
            for j in range(1,16):
                if sum < 10:
                    if row+j<=15:
                        sum+= data[row+j, col]  
                else:
                    j-=1
                    break              

            if sum == 10:
                if cnt<100:
                    if j<num+1:
                        if tmp<1+num or tmp>9-num:
                            draw_rectangle(add[row, col, 0], add[row, col, 1], 50 + add[row+ j, col , 0] - add[row, col, 0], 50 + add[row+ j, col, 1] - add[row, col, 1])
                            for i in range(j+1):                    
                                data[row+i, col]=0
                            found=1
                            break
                else:
                    draw_rectangle(add[row, col, 0], add[row, col, 1], 50 + add[row+ j, col , 0] - add[row, col, 0], 50 + add[row+ j, col, 1] - add[row, col, 1])
                    for i in range(j+1):                    
                        data[row+i, col]=0
                    found=1
                    break
        if(found):
            break
    return found

def process_data_rowcol(data, add):
    fault=0
    sum=0
    tmp=0
    found=0
    
    for row in range(16): 
        for col in range(10):       
            sum = data[row, col]
            if sum==0:
                continue
            tmp=data[row, col]
            for j in range(1,16):
                if row+j<=15:
                    sum+= data[row+j, col]
                    if sum==10:
                        draw_rectangle(add[row, col, 0], add[row, col, 1], 50 + add[row+ j, col , 0] - add[row, col, 0], 50 + add[row+ j, col, 1] - add[row, col, 1])
                        for a in range(j+1):                    
                            data[row+a, col]=0
                        found=1
                        break
                    elif sum>10:
                        fault=1
                        break
            sum = data[row, col]
            for i in range(1,10):
                if col+i<=9:
                    sum+= data[row, col+i]
                    if sum==10:
                        draw_rectangle(add[row, col, 0], add[row, col, 1], 50 + add[row, col + i, 0] - add[row, col, 0], 50 + add[row, col + i, 1] - add[row, col, 1])
                        for a in range(i+1):                    
                            data[row, col+a]=0
                        found=1
                        break
                    elif sum>10:
                        fault=1
                        break
            if(fault):
                fault=0
                continue       
            if(found):
                break
        if(found):
            break
    return found

def process_toadd(a,b,c,d):
    sum=0
    # sum=data[a,b]
    
    for i in range(c-a+1):     
        for j in range(d-b+1):     
            sum+=data[a+i,b+j]
    return sum

def process_toclear(a,b,c,d):
    sum=0
    # sum=data[a,b]
    
    for i in range(c-a+1):     
        for j in range(d-b+1):     
            data[a+i,b+j]=0
    return sum

def process_data_rowcol2(data, add):
    fault=0
    sum=0    
    for row in range(16): 
        for col in range(10):  
            sum = data[row, col]
            # if sum==0:
            #     continue    
            for i in range(row,row+20): 
                for j in range(col+1,col+20): 
                    if i<16 and j<10:                                          
                        sum = process_toadd(row,col,i,j)

                        if sum!=10:
                            continue
                        # tmp=data[i, j]
                        # if tmp==0:
                        #     continue
                        if sum==10:
                            draw_rectangle(30+add[row, col, 0], add[row, col, 1], 50 + add[i, j, 0] - add[row, col, 0], 50 + add[i, j, 1] - add[row, col, 1])
                            process_toclear(row,col,i,j)

# 算法，合10




# cnt=200
# f=process_data_rowcol(data, add)  
# for row in range(16):
#     for col in range(10):
#         print(int(data[row, col]), end='')
#     print("")
# cv2.waitKey(0)
while(1):    
    # f=1
    # while(f):
    #     f=process_data_row(data, add)
    #     cnt+=1
    # f=1    
    # while(f):
    #     f=process_data_col(data, add)
    #     cnt+=1
    f=1    
    while(f):
        f=process_data_rowcol(data, add)
    cnt+=1

    # f=1
    # # while(f):
    # process_data_rowcol2(data, add)        
    # # cv2.waitKey(0)
    # cnt+=1

    

#     if(cnt==40):
#         num+=1
#     if(cnt==60):
#         num+=1
#     if(cnt==80):
#         num+=1
    if(cnt>200):
        f=1    
        # while(f):
        process_data_rowcol2(data, add)
        
    if(cnt>300):
        break
    # print(cnt)


for row in range(16):
    for col in range(10):
        print(int(data[row, col]), end='')
    print("")

while(1):
    if keyboard.is_pressed('`'):
        f=process_data_row(data, add)
        f=process_data_col(data, add)
# # # draw_rectangle(X1, Y1, W1, H1)
