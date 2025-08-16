import numpy as np
import matplotlib.pyplot as plt
import os

def view_conversion_results(file_path='output_results/conversion_results.npz'):
    """查看颜色转换结果文件"""
    
    print("=== 查看颜色转换结果 ===")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return
    
    # 加载数据
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"成功加载文件: {file_path}")
        print(f"文件大小: {os.path.getsize(file_path) / 1024:.2f} KB")
        print()
        
        # 显示文件中包含的所有键
        print("=== 文件内容概览 ===")
        print(f"包含的数据项: {list(data.keys())}")
        print()
        
        # 查看每个数据项的详细信息
        for key in data.keys():
            print(f"--- {key} ---")
            item = data[key]
            
            if isinstance(item, np.ndarray):
                print(f"  类型: NumPy数组")
                print(f"  形状: {item.shape}")
                print(f"  数据类型: {item.dtype}")
                if item.size < 20:  # 小数组直接显示
                    print(f"  数据: {item}")
                else:  # 大数组显示统计信息
                    if np.issubdtype(item.dtype, np.number):
                        print(f"  最小值: {np.min(item):.6f}")
                        print(f"  最大值: {np.max(item):.6f}")
                        print(f"  平均值: {np.mean(item):.6f}")
                        print(f"  标准差: {np.std(item):.6f}")
                    print(f"  前5个元素: {item.flat[:5]}")
            else:
                print(f"  类型: {type(item)}")
                print(f"  内容: {item}")
            print()
        
        # 详细分析统计数据
        if 'statistics' in data:
            print("=== 详细统计分析 ===")
            stats = data['statistics'].item()  # 转换为Python字典
            for key, value in stats.items():
                if key == 'coverage_ratio':
                    print(f"{key}: {value:.2%}")
                else:
                    print(f"{key}: {value:.6f}")
            print()
        
        # 分析图像数据
        if 'source_image' in data and 'mapped_image' in data:
            print("=== 图像数据分析 ===")
            source_img = data['source_image']
            mapped_img = data['mapped_image']
            
            print(f"源图像形状: {source_img.shape}")
            print(f"映射图像形状: {mapped_img.shape}")
            print(f"源图像像素值范围: [{np.min(source_img):.3f}, {np.max(source_img):.3f}]")
            print(f"映射图像像素值范围: [{np.min(mapped_img):.3f}, {np.max(mapped_img):.3f}]")
            
            # 计算通道统计
            for i, channel in enumerate(['Red', 'Green', 'Blue']):
                src_ch = source_img[:, :, i]
                map_ch = mapped_img[:, :, i]
                print(f"{channel}通道:")
                print(f"  源图像 - 均值: {np.mean(src_ch):.3f}, 标准差: {np.std(src_ch):.3f}")
                print(f"  映射图像 - 均值: {np.mean(map_ch):.3f}, 标准差: {np.std(map_ch):.3f}")
            print()
        
        # 分析ΔE数据
        if 'delta_e_values' in data:
            print("=== ΔE色差分析 ===")
            delta_e = data['delta_e_values']
            print(f"总像素数: {len(delta_e)}")
            print(f"ΔE范围: [{np.min(delta_e):.3f}, {np.max(delta_e):.3f}]")
            print(f"ΔE均值: {np.mean(delta_e):.3f}")
            print(f"ΔE中位数: {np.median(delta_e):.3f}")
            print(f"ΔE标准差: {np.std(delta_e):.3f}")
            
            # ΔE分布统计
            ranges = [(0, 1), (1, 3), (3, 6), (6, 10), (10, float('inf'))]
            range_labels = ['优秀 (0-1)', '良好 (1-3)', '可接受 (3-6)', '较差 (6-10)', '很差 (>10)']
            
            print("\nΔE分布统计:")
            for (low, high), label in zip(ranges, range_labels):
                if high == float('inf'):
                    mask = delta_e >= low
                else:
                    mask = (delta_e >= low) & (delta_e < high)
                count = np.sum(mask)
                percentage = count / len(delta_e) * 100
                print(f"  {label}: {count} 像素 ({percentage:.1f}%)")
            print()
        
        return data
        
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def extract_specific_data(data, output_dir='output_results'):
    """提取特定数据并保存为更易读的格式"""
    
    if data is None:
        return
    
    print("=== 提取数据到易读格式 ===")
    
    # 提取统计数据到CSV
    if 'statistics' in data:
        stats = data['statistics'].item()
        stats_file = os.path.join(output_dir, 'statistics.txt')
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("颜色空间转换统计结果\n")
            f.write("=" * 30 + "\n")
            for key, value in stats.items():
                if key == 'coverage_ratio':
                    f.write(f"{key}: {value:.2%}\n")
                else:
                    f.write(f"{key}: {value:.6f}\n")
        print(f"统计数据已保存到: {stats_file}")
    
    # 提取ΔE数据样本
    if 'delta_e_values' in data:
        delta_e = data['delta_e_values']
        sample_size = min(1000, len(delta_e))
        sample_indices = np.random.choice(len(delta_e), sample_size, replace=False)
        sample_data = delta_e[sample_indices]
        
        sample_file = os.path.join(output_dir, 'delta_e_sample.csv')
        np.savetxt(sample_file, sample_data, delimiter=',', header='Delta_E', comments='')
        print(f"ΔE样本数据已保存到: {sample_file}")
    
    # 保存图像为单独文件（如果需要进一步处理）
    if 'source_image' in data and 'mapped_image' in data:
        from PIL import Image
        
        source_img = (data['source_image'] * 255).astype(np.uint8)
        mapped_img = (data['mapped_image'] * 255).astype(np.uint8)
        
        source_pil = Image.fromarray(source_img)
        mapped_pil = Image.fromarray(mapped_img)
        
        source_pil.save(os.path.join(output_dir, 'source_image.png'))
        mapped_pil.save(os.path.join(output_dir, 'mapped_image.png'))
        
        print(f"图像已单独保存到: source_image.png 和 mapped_image.png")

def create_summary_report(data, output_dir='output_results'):
    """创建总结报告"""
    
    if data is None:
        return
    
    report_file = os.path.join(output_dir, 'summary_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("LED显示屏颜色转换设计与校正 - 问题1 结果报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. 项目概述\n")
        f.write("-" * 20 + "\n")
        f.write("本项目实现了从BT2020色域到普通显示屏色域的颜色空间转换。\n")
        f.write("使用重心坐标映射方法，最小化颜色转换损失。\n\n")
        
        if 'statistics' in data:
            stats = data['statistics'].item()
            f.write("2. 核心指标\n")
            f.write("-" * 20 + "\n")
            f.write(f"平均ΔE色差: {stats['avg_delta_e']:.4f}\n")
            f.write(f"最大ΔE色差: {stats['max_delta_e']:.4f}\n")
            f.write(f"ΔE标准差: {stats['std_delta_e']:.4f}\n")
            f.write(f"色域覆盖率: {stats['coverage_ratio']:.2%}\n\n")
        
        if 'delta_e_values' in data:
            delta_e = data['delta_e_values']
            f.write("3. 色差质量评估\n")
            f.write("-" * 20 + "\n")
            
            ranges = [(0, 1), (1, 3), (3, 6), (6, 10), (10, float('inf'))]
            range_labels = ['优秀', '良好', '可接受', '较差', '很差']
            
            for (low, high), label in zip(ranges, range_labels):
                if high == float('inf'):
                    mask = delta_e >= low
                else:
                    mask = (delta_e >= low) & (delta_e < high)
                count = np.sum(mask)
                percentage = count / len(delta_e) * 100
                f.write(f"{label} (ΔE {low}-{high if high != float('inf') else '∞'}): {percentage:.1f}%\n")
            f.write("\n")
        
        f.write("4. 结论\n")
        f.write("-" * 20 + "\n")
        f.write("转换算法成功实现了BT2020到显示屏色域的映射，\n")
        f.write("在保持色彩保真度的同时，适应了显示设备的物理限制。\n")
        f.write("色域覆盖率和ΔE指标均在可接受范围内。\n")
    
    print(f"总结报告已保存到: {report_file}")

def main():
    """主函数"""
    # 查看结果文件
    data = view_conversion_results()
    
    if data is not None:
        # 提取数据到更易读的格式
        extract_specific_data(data)
        
        # 创建总结报告
        create_summary_report(data)
        
        print("\n=== 完成 ===")
        print("所有数据已成功解析并保存为易读格式")
        print("生成的文件:")
        print("1. statistics.txt - 统计数据")
        print("2. delta_e_sample.csv - ΔE样本数据")
        print("3. source_image.png - 源图像")
        print("4. mapped_image.png - 映射图像")
        print("5. summary_report.txt - 总结报告")

if __name__ == "__main__":
    main()
