import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colormath.color_objects import sRGBColor, XYZColor
from colormath.color_conversions import convert_color

# 设置matplotlib显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def rgb_to_xyz(r, g, b):
    # RGB转XYZ颜色空间
    color_rgb = sRGBColor(r, g, b, is_upscaled=True)
    color_xyz = convert_color(color_rgb, XYZColor)
    return color_xyz.xyz_x, color_xyz.xyz_y, color_xyz.xyz_z

def xyz_to_rgb(x, y, z):
    # XYZ转RGB颜色空间
    color_xyz = XYZColor(x, y, z)
    color_rgb = convert_color(color_xyz, sRGBColor)
    r = np.clip(color_rgb.rgb_r, 0, 1)
    g = np.clip(color_rgb.rgb_g, 0, 1)
    b = np.clip(color_rgb.rgb_b, 0, 1)
    return np.uint8(r * 255), np.uint8(g * 255), np.uint8(b * 255)

def color_diff(rgb1, rgb2):
    # 计算两个RGB颜色的欧氏距离，衡量色差
    return np.linalg.norm(np.array(rgb1) - np.array(rgb2))

def load_data(path):
    # 加载Excel文件中的各个sheet数据
    try:
        data = {
            'target': pd.read_excel(path, sheet_name='RGB目标值'),
            'R_R': pd.read_excel(path, sheet_name='R_R'),
            'R_G': pd.read_excel(path, sheet_name='R_G'),
            'R_B': pd.read_excel(path, sheet_name='R_B'),
            'G_R': pd.read_excel(path, sheet_name='G_R'),
            'G_G': pd.read_excel(path, sheet_name='G_G'),
            'G_B': pd.read_excel(path, sheet_name='G_B'),
            'B_R': pd.read_excel(path, sheet_name='B_R'),
            'B_G': pd.read_excel(path, sheet_name='B_G'),
            'B_B': pd.read_excel(path, sheet_name='B_B')
        }
        print("数据加载成功")
        return data
    except Exception as e:
        print(f"加载失败：{e}")
        return None

def get_min_shape(data, keys):
    """获取一组sheet的最小公共shape，保证拼接时不会越界"""
    shapes = [data[k].shape for k in keys]
    min_rows = min(s[0] for s in shapes)
    min_cols = min(s[1] for s in shapes)
    return min_rows, min_cols

def build_rgb_array(data, keys, rows, cols):
    """构建RGB三通道数组，自动裁剪为最小shape"""
    arr = np.zeros((rows, cols, 3), dtype=np.uint8)
    for idx, k in enumerate(keys):
        arr[..., idx] = data[k].values[:rows, :cols]
    return arr

def color_correction(data):
    # 取每组的最小shape，保证三通道shape一致
    r_rows, r_cols = get_min_shape(data, ['R_R', 'R_G', 'R_B'])
    g_rows, g_cols = get_min_shape(data, ['G_R', 'G_G', 'G_B'])
    b_rows, b_cols = get_min_shape(data, ['B_R', 'B_G', 'B_B'])
    rows = min(r_rows, g_rows, b_rows)
    cols = min(r_cols, g_cols, b_cols)

    # 构造原始RGB数据矩阵
    raw_R = build_rgb_array(data, ['R_R', 'R_G', 'R_B'], rows, cols)
    raw_G = build_rgb_array(data, ['G_R', 'G_G', 'G_B'], rows, cols)
    raw_B = build_rgb_array(data, ['B_R', 'B_G', 'B_B'], rows, cols)

    # 目标颜色（理想RGB值）
    target_R = (220, 0, 0)
    target_G = (0, 220, 0)
    target_B = (0, 0, 220)

    # 目标颜色的XYZ值
    xyz_R = rgb_to_xyz(*target_R)
    xyz_G = rgb_to_xyz(*target_G)
    xyz_B = rgb_to_xyz(*target_B)

    # 结果容器，存储校正后的RGB
    corrected_R = np.zeros((rows, cols, 3), dtype=np.uint8)
    corrected_G = np.zeros((rows, cols, 3), dtype=np.uint8)
    corrected_B = np.zeros((rows, cols, 3), dtype=np.uint8)

    # 校正红色通道
    for i in range(rows):
        for j in range(cols):
            orig_rgb = raw_R[i, j]
            orig_xyz = rgb_to_xyz(*orig_rgb)
            # 计算校正比例（将原始xyz调整到目标xyz）
            scale = [t / o if o > 1e-5 else 1.0 for t, o in zip(xyz_R, orig_xyz)]
            corrected_xyz = [o * s for o, s in zip(orig_xyz, scale)]
            corrected_R[i, j] = xyz_to_rgb(*corrected_xyz)
    # 校正绿色通道
    for i in range(rows):
        for j in range(cols):
            orig_rgb = raw_G[i, j]
            orig_xyz = rgb_to_xyz(*orig_rgb)
            scale = [t / o if o > 1e-5 else 1.0 for t, o in zip(xyz_G, orig_xyz)]
            corrected_xyz = [o * s for o, s in zip(orig_xyz, scale)]
            corrected_G[i, j] = xyz_to_rgb(*corrected_xyz)
    # 校正蓝色通道
    for i in range(rows):
        for j in range(cols):
            orig_rgb = raw_B[i, j]
            orig_xyz = rgb_to_xyz(*orig_rgb)
            scale = [t / o if o > 1e-5 else 1.0 for t, o in zip(xyz_B, orig_xyz)]
            corrected_xyz = [o * s for o, s in zip(orig_xyz, scale)]
            corrected_B[i, j] = xyz_to_rgb(*corrected_xyz)

    return raw_R, raw_G, raw_B, corrected_R, corrected_G, corrected_B

def show_images(before_R, before_G, before_B, after_R, after_G, after_B):
    # 可视化校正前后RGB图像
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].imshow(before_R)
    axs[0, 0].set_title("原始红色")
    axs[0, 1].imshow(before_G)
    axs[0, 1].set_title("原始绿色")
    axs[0, 2].imshow(before_B)
    axs[0, 2].set_title("原始蓝色")

    axs[1, 0].imshow(after_R)
    axs[1, 0].set_title("校正红色")
    axs[1, 1].imshow(after_G)
    axs[1, 1].set_title("校正绿色")
    axs[1, 2].imshow(after_B)
    axs[1, 2].set_title("校正蓝色")

    plt.tight_layout()
    plt.savefig("第三问校正图像展示.png")
    plt.show()

def calculate_uniformity(original_data, corrected_data, target_color):
    """计算颜色均匀性指标：颜色偏差的标准差、均值、最大值"""
    orig_flat = original_data.reshape(-1, 3)
    corr_flat = corrected_data.reshape(-1, 3)
    orig_diffs = [color_diff(rgb, target_color) for rgb in orig_flat]
    corr_diffs = [color_diff(rgb, target_color) for rgb in corr_flat]
    orig_std = np.std(orig_diffs)
    corr_std = np.std(corr_diffs)
    orig_mean = np.mean(orig_diffs)
    corr_mean = np.mean(corr_diffs)
    orig_max = np.max(orig_diffs)
    corr_max = np.max(corr_diffs)
    return orig_std, corr_std, orig_mean, corr_mean, orig_max, corr_max

def save_numerical_results(original_red, original_green, original_blue,
                          corrected_red, corrected_green, corrected_blue,
                          target_red, target_green, target_blue,
                          filename='颜色校正结果.txt'):
    """保存颜色校正的详细数值结果到文本文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("颜色校正结果分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        # 红色通道分析
        f.write("红色通道分析:\n")
        f.write("-" * 30 + "\n")
        orig_red_std, corr_red_std, orig_red_mean, corr_red_mean, orig_red_max, corr_red_max = calculate_uniformity(original_red, corrected_red, target_red)
        f.write(f"原始标准差: {orig_red_std:.2f}\n")
        f.write(f"校正后标准差: {corr_red_std:.2f}\n")
        f.write(f"改善百分比: {((orig_red_std - corr_red_std) / orig_red_std * 100):.2f}%\n")
        f.write(f"原始平均偏差: {orig_red_mean:.2f}\n")
        f.write(f"校正后平均偏差: {corr_red_mean:.2f}\n")
        f.write(f"原始最大偏差: {orig_red_max:.2f}\n")
        f.write(f"校正后最大偏差: {corr_red_max:.2f}\n\n")
        
        # 绿色通道分析
        f.write("绿色通道分析:\n")
        f.write("-" * 30 + "\n")
        orig_green_std, corr_green_std, orig_green_mean, corr_green_mean, orig_green_max, corr_green_max = calculate_uniformity(original_green, corrected_green, target_green)
        f.write(f"原始标准差: {orig_green_std:.2f}\n")
        f.write(f"校正后标准差: {corr_green_std:.2f}\n")
        f.write(f"改善百分比: {((orig_green_std - corr_green_std) / orig_green_std * 100):.2f}%\n")
        f.write(f"原始平均偏差: {orig_green_mean:.2f}\n")
        f.write(f"校正后平均偏差: {corr_green_mean:.2f}\n")
        f.write(f"原始最大偏差: {orig_green_max:.2f}\n")
        f.write(f"校正后最大偏差: {corr_green_max:.2f}\n\n")
        
        # 蓝色通道分析
        f.write("蓝色通道分析:\n")
        f.write("-" * 30 + "\n")
        orig_blue_std, corr_blue_std, orig_blue_mean, corr_blue_mean, orig_blue_max, corr_blue_max = calculate_uniformity(original_blue, corrected_blue, target_blue)
        f.write(f"原始标准差: {orig_blue_std:.2f}\n")
        f.write(f"校正后标准差: {corr_blue_std:.2f}\n")
        f.write(f"改善百分比: {((orig_blue_std - corr_blue_std) / orig_blue_std * 100):.2f}%\n")
        f.write(f"原始平均偏差: {orig_blue_mean:.2f}\n")
        f.write(f"校正后平均偏差: {corr_blue_mean:.2f}\n")
        f.write(f"原始最大偏差: {orig_blue_max:.2f}\n")
        f.write(f"校正后最大偏差: {corr_blue_max:.2f}\n\n")
        
        # 总体分析
        f.write("总体分析:\n")
        f.write("-" * 30 + "\n")
        total_orig_std = (orig_red_std + orig_green_std + orig_blue_std) / 3
        total_corr_std = (corr_red_std + corr_green_std + corr_blue_std) / 3
        f.write(f"总体原始标准差: {total_orig_std:.2f}\n")
        f.write(f"总体校正后标准差: {total_corr_std:.2f}\n")
        f.write(f"总体改善百分比: {((total_orig_std - total_corr_std) / total_orig_std * 100):.2f}%\n")

def main():
    file = 'RGB数值.xlsx'
    data = load_data(file)
    if not data:
        return

    # 颜色校正，得到校正前后RGB矩阵
    raw_R, raw_G, raw_B, corrected_R, corrected_G, corrected_B = color_correction(data)

    # 可视化校正前后效果
    show_images(raw_R, raw_G, raw_B, corrected_R, corrected_G, corrected_B)

    # 目标RGB值
    target_red = (220, 0, 0)
    target_green = (0, 220, 0)
    target_blue = (0, 0, 220)

    # 输出分析结果到屏幕
    orig_red_std, corr_red_std, orig_red_mean, corr_red_mean, orig_red_max, corr_red_max = calculate_uniformity(raw_R, corrected_R, target_red)
    orig_green_std, corr_green_std, orig_green_mean, corr_green_mean, orig_green_max, corr_green_max = calculate_uniformity(raw_G, corrected_G, target_green)
    orig_blue_std, corr_blue_std, orig_blue_mean, corr_blue_mean, orig_blue_max, corr_blue_max = calculate_uniformity(raw_B, corrected_B, target_blue)

    print("\n颜色校正结果分析:")
    print("=" * 50)
    print("\n红色通道:")
    print(f"原始标准差: {orig_red_std:.2f}")
    print(f"校正后标准差: {corr_red_std:.2f}")
    print(f"改善百分比: {((orig_red_std - corr_red_std) / orig_red_std * 100):.2f}%")
    print(f"原始平均偏差: {orig_red_mean:.2f}")
    print(f"校正后平均偏差: {corr_red_mean:.2f}")
    print(f"原始最大偏差: {orig_red_max:.2f}")
    print(f"校正后最大偏差: {corr_red_max:.2f}")

    print("\n绿色通道:")
    print(f"原始标准差: {orig_green_std:.2f}")
    print(f"校正后标准差: {corr_green_std:.2f}")
    print(f"改善百分比: {((orig_green_std - corr_green_std) / orig_green_std * 100):.2f}%")
    print(f"原始平均偏差: {orig_green_mean:.2f}")
    print(f"校正后平均偏差: {corr_green_mean:.2f}")
    print(f"原始最大偏差: {orig_green_max:.2f}")
    print(f"校正后最大偏差: {corr_green_max:.2f}")

    print("\n蓝色通道:")
    print(f"原始标准差: {orig_blue_std:.2f}")
    print(f"校正后标准差: {corr_blue_std:.2f}")
    print(f"改善百分比: {((orig_blue_std - corr_blue_std) / orig_blue_std * 100):.2f}%")
    print(f"原始平均偏差: {orig_blue_mean:.2f}")
    print(f"校正后平均偏差: {corr_blue_mean:.2f}")
    print(f"原始最大偏差: {orig_blue_max:.2f}")
    print(f"校正后最大偏差: {corr_blue_max:.2f}")

    # 计算总体改善
    total_orig_std = (orig_red_std + orig_green_std + orig_blue_std) / 3
    total_corr_std = (corr_red_std + corr_green_std + corr_blue_std) / 3
    print("\n总体分析:")
    print(f"总体原始标准差: {total_orig_std:.2f}")
    print(f"总体校正后标准差: {total_corr_std:.2f}")
    print(f"总体改善百分比: {((total_orig_std - total_corr_std) / total_orig_std * 100):.2f}%")

    # 保存数值结果到文本文件
    save_numerical_results(raw_R, raw_G, raw_B,
                          corrected_R, corrected_G, corrected_B,
                          target_red, target_green, target_blue)

    # 保存校正后RGB到Excel
    with pd.ExcelWriter("第三问-校正后RGB.xlsx") as writer:
        for mat, name in zip([corrected_R, corrected_G, corrected_B], ['R', 'G', 'B']):
            pd.DataFrame(mat[:, :, 0]).to_excel(writer, sheet_name=f'{name}_R', index=False)
            pd.DataFrame(mat[:, :, 1]).to_excel(writer, sheet_name=f'{name}_G', index=False)
            pd.DataFrame(mat[:, :, 2]).to_excel(writer, sheet_name=f'{name}_B', index=False)

    print("校正完成，结果保存至 Excel 和图像")

if __name__ == '__main__':
    main()

# 代码说明：
# 本代码用于第三问的色彩校正建模。主要流程为：
# 1. 读取Excel中各通道的原始RGB数据。
# 2. 将RGB转换到XYZ空间，计算校正比例，使其更接近目标色。
# 3. 校正后再转回RGB，得到校正后的三通道图像。
# 4. 对校正前后结果进行可视化和数值分析，输出改善效果。
# 5. 保存校正后数据和分析结果，便于后续报告撰写和复现。
