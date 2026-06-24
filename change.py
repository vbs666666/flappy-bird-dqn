import cv2
import numpy as np

# 读取图像
image = cv2.imread('./assets/sprites/redbird-upflap1.png', cv2.IMREAD_UNCHANGED)

# 如果图像没有alpha通道，添加一个
if image.shape[2] == 3:  # 如果只有RGB三个通道
    image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

# 缩放图像到34x24像素
image = cv2.resize(image, (34, 24), interpolation=cv2.INTER_LANCZOS4)

# 获取背景色（假设左上角的颜色为背景色）
background_color = image[0, 0]

# 创建一个与图像相同大小的透明图像
transparent_image = np.zeros((24, 34, 4), dtype=np.uint8)

# 遍历图像数据并设置背景为透明
for y in range(24):
    for x in range(34):
        if np.array_equal(image[y, x][:3], background_color[:3]):
            transparent_image[y, x] = [0, 0, 0, 0]  # 设置为全透明
        else:
            transparent_image[y, x] = image[y, x]

# 保存结果图像
cv2.imwrite('./assets/sprites/redbird-upflap.png', transparent_image)
