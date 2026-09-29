import cv2
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

'''
读取：
           cv.imread(path, flag=1) 
                flag=1: BGR三通道图像数据
                flag=0：灰度图像
                flag=-1：保持图像所有通道进行读取
                高 宽 通道
                h,w,c-BGR-->h,w,RGB
        展示：
            cv.imshow("窗口名称", data)
            cv.waitKey([number]): 展示窗口中的图像，设置窗口中的图像刷新的间隔时间。0和没有数字需要使用键盘的任意键刷新

        保存：
            cv.imwrite(path, data): 将图像数据保存到新的图像中
'''

# def imread(path, flag=1):
"""
    读取图像文件
    Args:
        path (str): 图像文件路径
        flag (int): 读取模式，1=BGR三通道，0=灰度图，-1=原生通道
    Returns:
        numpy.ndarray: 图像数据数组，形状为(高度, 宽度, 通道数)
"""
# return cv.imread(path, flag)


img1 = cv.imread(r"../aitest/kdqs.jpg")
img2 = cv.imread(r"../aitest/xqs.jpg")
# plt.imshow(img1[:, :, ::1])
# plt.show()
print(img1.shape)  # (685, 700, 3)
print(img2.shape)  # (3088, 1456, 3)
#
# cv.imshow("xxx", img1)
# cv.waitKey(50)

'''
图像数据编辑：
        cv.line(data, pt1=开始点, pt2=结束点, 颜色=color, 线的粗细=thickness)：直线
        cv.rectangle(data, pt1=左上角坐标, pt2=右下角坐标, 颜色=color, 线的粗细=thickness)：矩形，thickness=-1实心矩形
        cv.circle(data, pt=中心点坐标, r=半径,  颜色=color, 线的粗细=thickness)：圆形，thickness=-1实心圆
        cv.putText(data, 文本=text, 文本的开始点=pt, 字体=font, 字体大小=fontSize, 颜色=color, 线的粗细=thickness, lineType=cv.LINE_AA-抗锯齿)：写入文字

'''


def draw_shapes(img):
    """
    在图像上绘制图形示例
    Args:
        img (numpy.ndarray): 原始图像数据
    """
    # 绘制直线：从坐标(50,50)到(200,200)，颜色为BGR格式的(235,148,109)，线宽2像素
    cv.line(img, pt1=(50, 50), pt2=(200, 200), color=(235, 148, 109), thickness=2)
    # 绘制矩形：左上角(50,50)到右下角(200,200)，颜色和线宽与直线相同
    cv.rectangle(img, pt1=(50, 50), pt2=(200, 200), color=(235, 148, 109), thickness=2)
    # 绘制圆形：中心点(125,125)，半径75像素，颜色和线宽保持一致
    cv.circle(img, center=(125, 125), radius=75, color=(235, 148, 109), thickness=2)
    # 添加文字：内容为"hello"，起始坐标(50,50)，使用复杂字体样式，字号2，颜色与线宽相同，启用抗锯齿处理
    cv.putText(img, text="hello", org=(50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(235, 148, 109),
               thickness=2, lineType=cv2.LINE_AA)
    # 将绘制结果保存到指定路径的图像文件
    cv.imwrite(r"../aitest/kdqs1.jpg", img)
    # 创建窗口显示绘制后的图像，等待用户按键后继续执行
    cv.imshow("xx", img)
    cv.waitKey()


'''
图像融合：图像数据的形状必须一致
     +
     cv.add(img1, img2)
     cv.addWeighted(img1, alpha, img2, beta, gamma)
        res = alpha * img1 + beta * img2 + gamma
'''

# 融合
# [高, 宽, 通道] 图像的裁剪 对图片进行切片
# img2 = img2[500:1185, 1760:2460, :]
# # resize() 将图片直接压缩到指定尺寸(宽, 高)
# img2 = cv.resize(img2, (700, 685), fx=0, fy=0, interpolation=cv.INTER_LINEAR)
# res = cv.add(img1, img2)
# # res = img2 + img1
# cv.imshow("xx", img2)
# cv.imshow("xx", res)
# cv.waitKey(0)
#
# res = cv.addWeighted(img1, 0.5, img2, 0.7, 50)
# cv.imshow("xx", res)
# cv.imshow("q", img1)
# cv.waitKey(0)


'''
几何变换：
        cv.resize(data, dsize=绝对尺寸, fx=相对尺寸宽的比例, fy=相对尺寸高的比例, interpolation=cv.INTER_LINEAR)：对图像进行缩放
                    使用绝对尺寸时，不要使用相对尺寸
                    使用相对尺寸，不要使用绝对尺寸
                    interpolation: 表示插值的方式，一般是线性插值

        cv.warpAffine(data, M, (w, h)): 根据M矩阵实现图形的几何变换
            M = np.float32([[1, 0, x], [0, 1, y]]): 表示水平和垂直移动多少
            M = cv.getRotationMatrix2D(pt=旋转点, angle=角度, scale=缩放比例)：获取旋转矩阵
            M = cv.getAffineTransform(pt1=原始图像上的三个点, pt2=新平面上的三个点)

        M = cv.getPerspectiveTransform(pt1=原始图像上的四个点, pt2=新平面上的四个点)
        cv.warpPerspective(data, M, (w, h)): 实现图像的投射变换
'''


def geometric_transformations(img):
    """
    执行图像几何变换示例，包含平移、旋转、仿射变换和透视变换
    Args:
        img (numpy.ndarray): 原始图像数据
    """
    h, w, c = img.shape
    # 创建平移矩阵：向右平移100像素，向下平移50像素
    m = np.float32([[1, 0, 100], [0, 1, 50]])
    res = cv.warpAffine(img, m, (w, h))
    cv.imshow("zzc", res)
    cv.waitKey(50)

    # 动态平移演示：水平平移范围0-90像素，垂直平移范围0-90像素
    for i in range(90):
        m = np.float32([[1, 0, 100 + i], [0, 1, 50 + 1 * i]])
        res = cv.warpAffine(img, m, (w, h))
        cv.imshow("1", res)
        cv.waitKey(50)

    # 创建旋转矩阵：以(300,300)为中心点旋转90度，不缩放
    m = cv.getRotationMatrix2D(center=(300, 300), angle=90, scale=1)
    res = cv.warpAffine(img, m, (w, h))
    cv.imshow("2", res)
    cv.waitKey(50)

    # 动态旋转演示：0-90度渐进旋转，同时进行0-1的线性缩放
    for i in range(90):
        m = cv.getRotationMatrix2D(center=(300, 300), angle=i, scale=1 - 0.005 * i)
        res = cv.warpAffine(img, m, (w, h))
        cv.imshow("2", res)
        cv.waitKey(50)

    # 创建仿射变换矩阵：将源三角形[[50,50],[100,100],[200,300]]映射到目标三角形
    m = cv.getAffineTransform(np.float32([[50, 50], [100, 100], [200, 300]]),
                              np.float32([[50, 80], [120, 100], [250, 300]]))
    res = cv.warpAffine(img, m, (w, h))
    cv.imshow("3", res)
    cv.waitKey(50)

    # 动态仿射变换演示：动态调整源三角形和目标三角形坐标参数
    for i in range(90):
        m = cv.getAffineTransform(np.float32([[50, 50], [100, 100], [200, 300 + 2 * i]]),
                                  np.float32([[50, 80 + i], [120, 100], [250 + i, 300 - i]]))
        res = cv.warpAffine(img, m, (w, h))
        cv.imshow("3", res)
        cv.waitKey(50)

        # 创建透视变换矩阵：将源四边形映射到目标四边形
        m = cv.getPerspectiveTransform(np.float32([[50, 50], [100, 100], [200, 300], [300, 350]]),
                                       np.float32([[50, 50], [100, 100], [200, 300], [310, 350]]))
        res = cv.warpPerspective(img, m, (w, h))
        cv.imshow("4", res)
        cv.waitKey(50)

    # 动态透视变换演示：渐进式调整目标四边形坐标参数
    for i in range(90):
        m = cv.getPerspectiveTransform(np.float32([[50, 50], [100, 100], [200, 300], [300, 350]]),
                                       np.float32([[50, 50 + 0.1 * i], [100, 100], [200, 300],
                                                   [310 + 0.1 * i, 350 + 0.01 * i]]))
        res = cv.warpPerspective(img, m, (w, h))
        cv.imshow("4", res)
        cv.waitKey(50)


# 图像降噪
'''
图像的噪声：高斯噪声，椒盐噪声
        均值降噪：cv.blur(data, ksize=指定内核大小)
        高斯降噪: cv.GaussianBlur(data, ksize=指定内核大小, sigmaX=水平方向的标准差, sigmaY=垂直方向的标准差)
                        sigmaY=0沿用水平方向的标准差
        中值降噪：cv.medianBlur(data, ksize=指定内核大小)

'''


def denoise_image():
    """
    执行图像降噪处理对比演示
    Args:
        img (numpy.ndarray): 输入图像参数（实际未使用，函数内读取固定路径图像）
    """
    # 读取预存的带高斯噪声的狗狗测试图像
    img3 = cv.imread(r"../aitest/dogGauss.jpeg", 1)
    print(img3.shape)
    # cv.imshow("5", img3)
    # cv.waitKey(0)

    # 创建2行2列的子图布局，用于展示不同降噪效果对比
    fig, ax = plt.subplots(2, 2, figsize=(8, 5))
    # 原始图像展示
    ax[0, 0].imshow(img3)
    ax[0, 0].set_title("raw img")

    # 均值降噪处理（3x3方框滤波器）
    new1_img3 = cv.blur(img3, ksize=(3, 3))
    ax[0, 1].imshow(new1_img3)
    ax[0, 1].set_title("blur")

    # 高斯降噪处理（3x3高斯核，水平标准差1）
    new2_img3 = cv.GaussianBlur(img3, ksize=(3, 3), sigmaX=1, sigmaY=0)
    ax[1, 0].imshow(new2_img3)
    ax[1, 0].set_title("GaussianBlur")

    # 中值降噪处理（3x3邻域取中值）
    new3_img3 = cv.medianBlur(img3, ksize=3)
    ax[1, 1].imshow(new3_img3)
    ax[1, 1].set_title("medianBlur")

    # cv.imshow("6", new_img3)
    # cv.waitKey(0)
    # 统一显示所有降噪效果对比图像
    plt.show()


# 边缘检查
'''
边缘检查：
        一阶导数：Sobel，Scharr算子
            cv.Sobel(data, ddepth, ksize, dx, dy)
                ddepth:改变Sobel算子计算结构的数据类型。
                ksize：卷积核大小, 如果为-1就是scharr算子
                dx和dy表示对图像水平和垂直方向求的一阶导数
        二阶导数：Laplacian算子
            cv.Laplacian(data, ddepth, ksize): laplacian算子
            cv.convertScaleAbs(data): 获取数据的绝对值
        
        cv.Canny(data, thresholdMin, thresholdMax):
            1: 对图像进行高斯降噪
            2：使用sobel算子对图像进行边缘检查
            3：对Sobel的结果进行非极大值抑制(NMS), ksize范围内某个点是否为最大值，如果为最大值，就保留，如果不为最大值就置为0
            4：滞后阈值(minVal, maxVal)
               某个点是否超过maxVal，如果超过maxVal就表示边缘，如果某个在minVal和maxVal，就要看这个点是否和边缘相连，如果是，就保留，否则就抑制。
'''


def edge_detection():
    """
    执行边缘检测操作
    Args:
        img (numpy.ndarray): 输入图像数据（应为灰度图）
    Returns:
        Tuple[numpy.ndarray]: 返回不同边缘检测算法的结果
    """
    # Sobel边缘检测
    img4 = cv.imread(r"../aitest/img.jpg", 0)
    print(img4.shape)
    sobeldx_img4 = cv.Sobel(img4, cv.CV_16S, ksize=3, dx=1, dy=0)
    sobeldy_img4 = cv.Sobel(img4, cv.CV_16S, ksize=3, dx=0, dy=1)
    fig, ax = plt.subplots(2, 2, figsize=(8, 5))
    sobel_img4 = cv.convertScaleAbs(sobeldx_img4) + cv.convertScaleAbs(sobeldy_img4)
    ax[0, 0].imshow(sobel_img4)
    ax[0, 0].set_title("sobel")
    # cv.imshow("7", sobel_img4)
    # cv.waitKey(0)
    new_sobel_img4 = cv.addWeighted(cv.convertScaleAbs(sobeldx_img4), 0.5, cv.convertScaleAbs(sobeldy_img4), 0.5, 0)
    ax[0, 1].imshow(new_sobel_img4)
    ax[0, 1].set_title("new_sobel")
    # cv.imshow("8", new_sobel_img4)
    # cv.waitKey(0)

    # Laplacian边缘检测
    lpls_img4 = cv.Laplacian(img4, ddepth=cv.CV_16S, ksize=3)
    ax[1, 0].imshow(cv.convertScaleAbs(lpls_img4))
    ax[1, 0].set_title("laplacian")
    # cv.imshow("9", cv.convertScaleAbs(lpls_img4))
    # cv.waitKey(0)

    # Canny边缘检测
    canny_img4 = cv.Canny(img4, 40, 100)
    ax[1, 1].imshow(canny_img4)
    ax[1, 1].set_title("canny")
    plt.show()


# 视频读写
'''
视频读写：
        cap = cv.VideoCapture(path): 获取视频对象
        cap.isOpened(): 是否打开视频
        cap.get(属性id): 获取视频属性
        cap.set(属性id, value): 设置视频的属性值--一般并不设置
        state, frame = cap.read(): 读取视频的每一帧图像
                state: True读取视频图像成功, False读取失败
                frame：获取读取成功的图像数据

        writer = cv.VideoWriter(path, cv.VideoWriter_fourcc=视频的编码格式, fps=帧数, (w,h))：创建视频的写入对象
        writer.write(img):将图像数据写入到视频中

        release(): 资源回收
'''


def process_video():
    """
    视频处理函数：读取视频文件，添加文字并进行边缘检测处理
    Args:
        input_path (str): 输入视频文件路径
        output_path (str): 输出视频文件路径
    """
    cap = cv.VideoCapture(r"../aitest/DOG.wmv")
    print(cap.isOpened())
    # 宽, 高, fps
    print(cap.get(3), cap.get(4), cap.get(5))

    # 创建视频的写入对象
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter("../aitest/DOG.mp4", fourcc, 10, (int(cap.get(3)), int(cap.get(4))))

    i = 1
    while True:
        state, frame = cap.read()
        if not state:
            break
        # 添加动态文字水印
        cv.putText(frame, 'cute dog', (50 + i, 50 + i), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
        # 偶数帧执行边缘检测
        if i % 2 == 0:
            # 获取灰度图
            gray_img = frame[:, :, 0]
            # Canny边缘检查
            canny_img = cv.Canny(gray_img, 10, 200)
            # 升维处理
            canny_img_data = canny_img.reshape(canny_img.shape[0], canny_img.shape[1], 1)
            canny_img_res = np.concatenate((canny_img_data, canny_img_data, canny_img_data), axis=-1)
            # 将每帧图像写入到新的视频中
            writer.write(canny_img_res)
        else:
            # 将每帧图像写入到新的视频中
            writer.write(frame)

        # cv.imshow("1", frame)
        # cv.waitKey(5)

        i += 1

    cap.release()
    writer.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # draw_shapes(img1)
    # 几何变换演示
    # geometric_transformations(img1)

    # 降噪处理演示
    # denoise_image()

    # 边缘检测演示
    # edge_detection()

    # 视频处理演示
    process_video()
