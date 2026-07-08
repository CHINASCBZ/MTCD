import os
import cv2
import numpy as np

# 原始 BCD 文件夹
bcd_folder = './bcd'  # 修改为你的路径
# 保存可视化后的文件夹
output_folder = './bcd_visual'

os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹里的所有图片
for filename in os.listdir(bcd_folder):
    if filename.endswith(('.png', '.jpg', '.tif')):  # 根据文件类型调整
        filepath = os.path.join(bcd_folder, filename)
        # 以灰度方式读取
        bcd = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        # 将 0/1 映射到 0/255，提高可见性
        bcd_visual = (bcd * 255).astype(np.uint8)

        # 保存到新的文件夹
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, bcd_visual)

print(f'可视化 BCD 图像已保存到 {output_folder}')