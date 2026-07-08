import os
import cv2
import numpy as np

# 原始 curr_seg 文件夹
curr_seg_folder = './curr_seg'  # 修改为你的路径
# 保存可视化后的文件夹
output_folder = './curr_seg_visual'
os.makedirs(output_folder, exist_ok=True)

# 定义每个类别对应的颜色 (BGR 格式)
# colors = {
#     0: (128, 128, 128),  # Bareland - 灰色
#     1: (0, 0, 255),      # Water - 红色
#     2: (0, 255, 0),      # Building - 绿色
#     3: (255, 0, 0),      # Structure - 蓝色
#     4: (0, 255, 255),    # Farmland - 黄色
#     5: (0, 128, 0),      # Vegetation - 深绿色
#     6: (128, 64, 0)      # Road - 棕色
# }
#这个是和对比实验一样的颜色映射：
colors = {#(BGR 格式)
    0: (0, 128, 255),    # 0 Bareland 裸地：橙色，RGB(255,128,0) -> BGR(0,128,255)
    1: (255, 0, 0),      # 1 Water 水体：蓝色，RGB(0,0,255) -> BGR(255,0,0)
    2: (0, 0, 255),      # 2 Building 建筑：红色，RGB(255,0,0) -> BGR(0,0,255)
    3: (0, 255, 255),    # 3 Structure 构筑物：黄色，RGB(255,255,0) -> BGR(0,255,255)
    4: (0, 255, 0),      # 4 Farmland 耕地：亮绿，RGB(0,255,0) -> BGR(0,255,0)
    5: (0, 128, 0),      # 5 Vegetation 植被：深绿，RGB(0,128,0) -> BGR(0,128,0)
    6: (128, 128, 128),  # 6 Road 道路：灰色，RGB(128,128,128) -> BGR(128,128,128)
}
SC_SCD7_PALETTE = np.array([    ##(RGB 格式)
    (255, 128, 0),    # 0 Bareland裸地：橙色
    (0, 0, 255),      # 1 Water水体：蓝色
    (255, 0, 0),      # 2 Building建筑：红色
    (255, 255, 0),    # 3 Structure构筑物：黄色
    (0, 255, 0),      # 4 Farmland耕地：亮绿
    (0, 128, 0),      # 5 Vegetation植被：深绿
    (128, 128, 128),  # 6 Road道路：灰色
], dtype=np.uint8)

# 遍历文件夹
for filename in os.listdir(curr_seg_folder):
    if filename.endswith(('.png', '.jpg', '.tif')):
        filepath = os.path.join(curr_seg_folder, filename)
        seg = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        # 创建彩色图
        seg_color = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in colors.items():
            seg_color[seg == label] = color
        # 保存
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, seg_color)

print(f'彩色可视化 curr_seg 图像已保存到 {output_folder}')