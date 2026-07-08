import os
import cv2
import numpy as np

# ================== 路径 ==================
prev_seg_folder = './prev_seg'
curr_seg_folder = './curr_seg'
bcd_folder = './bcd'

output_t1_folder = './prev_change_visual'
output_t2_folder = './curr_change_visual'

os.makedirs(output_t1_folder, exist_ok=True)
os.makedirs(output_t2_folder, exist_ok=True)

# ================== 颜色映射（BGR）==================

colors = {#(BGR 格式)
    0: (0, 128, 255),    # 0 Bareland 裸地：橙色，RGB(255,128,0) -> BGR(0,128,255)
    1: (255, 0, 0),      # 1 Water 水体：蓝色，RGB(0,0,255) -> BGR(255,0,0)
    2: (0, 0, 255),      # 2 Building 建筑：红色，RGB(255,0,0) -> BGR(0,0,255)
    3: (0, 255, 255),    # 3 Structure 构筑物：黄色，RGB(255,255,0) -> BGR(0,255,255)
    4: (0, 255, 0),      # 4 Farmland 耕地：亮绿，RGB(0,255,0) -> BGR(0,255,0)
    5: (0, 128, 0),      # 5 Vegetation 植被：深绿，RGB(0,128,0) -> BGR(0,128,0)
    6: (128, 128, 128),  # 6 Road 道路：灰色，RGB(128,128,128) -> BGR(128,128,128)
}
# 白色（未变化区域）
WHITE = (255, 255, 255)

# ================== 遍历文件 ==================
for filename in os.listdir(curr_seg_folder):

    if not filename.endswith(('.png', '.jpg', '.tif')):
        continue

    prev_path = os.path.join(prev_seg_folder, filename)
    curr_path = os.path.join(curr_seg_folder, filename)
    bcd_path  = os.path.join(bcd_folder, filename)

    if not (os.path.exists(prev_path) and os.path.exists(bcd_path)):
        print(f"跳过 {filename}（缺少文件）")
        continue

    # ================== 读取 ==================
    prev_seg = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
    curr_seg = cv2.imread(curr_path, cv2.IMREAD_GRAYSCALE)
    bcd = cv2.imread(bcd_path, cv2.IMREAD_GRAYSCALE)

    if prev_seg is None or curr_seg is None or bcd is None:
        print(f"读取失败 {filename}")
        continue

    # ================== 对齐检查 ==================
    if prev_seg.shape != curr_seg.shape:
        print(f"尺寸不一致 {filename}")
        continue

    # ================== 变化掩码 ==================
    change_mask = (bcd == 1) & (prev_seg != curr_seg)
    h, w = prev_seg.shape

    # ================== 初始化输出 ==================
    t1_vis = np.full((h, w, 3), WHITE, dtype=np.uint8)
    t2_vis = np.full((h, w, 3), WHITE, dtype=np.uint8)

    # ================== 上色 ==================
    for label, color in colors.items():

        # T1
        mask_t1 = change_mask & (prev_seg == label)
        t1_vis[mask_t1] = color

        # T2
        mask_t2 = change_mask & (curr_seg == label)
        t2_vis[mask_t2] = color

    # ================== 保存 ==================
    cv2.imwrite(os.path.join(output_t1_folder, filename), t1_vis)
    cv2.imwrite(os.path.join(output_t2_folder, filename), t2_vis)

print("T1 / T2 变化可视化完成！")