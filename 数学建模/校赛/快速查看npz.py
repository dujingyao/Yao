import numpy as np

def quick_view_npz(file_path):
    """快速查看NPZ文件的简单方法"""
    
    # 加载文件
    data = np.load(file_path, allow_pickle=True)
    
    print(f"文件: {file_path}")
    print(f"包含的数据: {list(data.keys())}")
    print()
    
    # 显示每个数据项的基本信息
    for key in data.keys():
        item = data[key]
        if isinstance(item, np.ndarray):
            if item.dtype == 'object':  # 处理字典等对象
                print(f"{key}: {item.item()}")
            else:
                print(f"{key}: 形状={item.shape}, 类型={item.dtype}")
                if item.size <= 10:
                    print(f"  数据: {item}")
                else:
                    print(f"  范围: [{np.min(item):.3f}, {np.max(item):.3f}]")
        else:
            print(f"{key}: {item}")
        print()

# 使用示例
if __name__ == "__main__":
    quick_view_npz('output_results/conversion_results.npz')
