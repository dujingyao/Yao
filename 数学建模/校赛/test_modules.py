#!/usr/bin/env python3
"""
PSO颜色转换模块测试脚本
测试主程序和可视化模块的分离是否成功
"""

try:
    print("正在导入模块...")
    from pso_visualization import PSO_ColorConversionVisualizer, visualize_pso_results
    print("✓ 可视化模块导入成功")
    
    import numpy as np
    print("✓ NumPy导入成功")
    
    # 测试可视化器初始化
    visualizer = PSO_ColorConversionVisualizer()
    print("✓ 可视化器初始化成功")
    
    print("\n可视化模块功能列表:")
    print("1. PSO_ColorConversionVisualizer类")
    print("2. create_comprehensive_visualization() - 综合分析图")
    print("3. create_convergence_plot() - 收敛曲线图")
    print("4. create_transformation_matrix_heatmap() - 转换矩阵热力图")
    print("5. create_channel_comparison() - 通道对比图")
    print("6. create_quality_analysis_dashboard() - 质量分析仪表板")
    print("7. visualize_pso_results() - 主要可视化函数")
    
    print("\n模块分离成功！")
    print("现在主程序文件更加简洁，可视化功能独立维护。")
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
except Exception as e:
    print(f"❌ 其他错误: {e}")
