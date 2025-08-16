import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
import os

# 设置matplotlib后端和字体
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

class PSO_ColorConversionVisualizer:
    """PSO颜色转换结果可视化器"""
    
    def __init__(self):
        self.setup_style()
    
    def setup_style(self):
        """设置可视化样式"""
        self.colors = {
            'excellent': '#2E8B57',  # 深绿色
            'good': '#90EE90',       # 浅绿色
            'acceptable': '#FFD700',  # 金色
            'poor': '#FF6347'        # 番茄红
        }
        
        self.figure_size = (24, 20)
        self.dpi = 300
    
    def create_comprehensive_visualization(self, original_rgbv, converted_rgbcx, 
                                         converter, quality_metrics, 
                                         transformation_matrix, save_dir):
        """创建完整的PSO优化结果可视化"""
        
        fig = plt.figure(figsize=self.figure_size)
        
        # 创建图像对比区域
        self._create_image_comparison_section(fig, original_rgbv, converted_rgbcx, 
                                            transformation_matrix, quality_metrics)
        
        # 创建色域映射图
        self._create_gamut_mapping_section(fig, converter)
        
        # 创建统计分析区域
        self._create_statistics_section(fig, quality_metrics)
        
        # 创建算法说明区域
        self._create_algorithm_explanation_section(fig)
        
        # 设置总标题
        plt.suptitle('PSO-Optimized 4-Channel RGBV to 5-Channel RGBCX Conversion Analysis',
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 保存图像
        plt.savefig(os.path.join(save_dir, 'pso_optimized_conversion_analysis.png'),
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _create_image_comparison_section(self, fig, original_rgbv, converted_rgbcx, 
                                       transformation_matrix, quality_metrics):
        """创建图像对比区域"""
        gs1 = fig.add_gridspec(3, 4, left=0.05, right=0.70, top=0.95, bottom=0.55)
        
        # 原始RGBV图像 (RGB部分)
        ax1 = fig.add_subplot(gs1[0, 0])
        rgb_display = original_rgbv[:, :, :3]
        ax1.imshow(rgb_display)
        ax1.set_title("Original RGBV\n(RGB channels)", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # V通道显示
        ax2 = fig.add_subplot(gs1[0, 1])
        v_channel = original_rgbv[:, :, 3]
        im1 = ax2.imshow(v_channel, cmap='viridis')
        ax2.set_title("V Channel\n(Violet)", fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im1, ax=ax2, shrink=0.8)
        
        # 转换后RGBCX图像 (RGB部分)
        ax3 = fig.add_subplot(gs1[0, 2])
        rgb_converted = converted_rgbcx[:, :, :3]
        ax3.imshow(rgb_converted)
        ax3.set_title("Converted RGBCX\n(RGB channels)", fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # 转换矩阵可视化
        ax4 = fig.add_subplot(gs1[0, 3])
        im_matrix = ax4.imshow(transformation_matrix, cmap='RdBu', vmin=-1, vmax=1)
        ax4.set_title("Transformation Matrix\n(PSO Optimized)", fontsize=12, fontweight='bold')
        ax4.set_xlabel("Output Channels (RGBCX)")
        ax4.set_ylabel("Input Channels (RGBV)")
        ax4.set_xticks(range(5))
        ax4.set_xticklabels(['R', 'G', 'B', 'C', 'X'])
        ax4.set_yticks(range(4))
        ax4.set_yticklabels(['R', 'G', 'B', 'V'])
        plt.colorbar(im_matrix, ax=ax4, shrink=0.8)
        
        # C通道显示
        ax5 = fig.add_subplot(gs1[1, 0])
        c_channel = converted_rgbcx[:, :, 3]
        im2 = ax5.imshow(c_channel, cmap='cool')
        ax5.set_title("C Channel\n(Cyan)", fontsize=12, fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im2, ax=ax5, shrink=0.8)
        
        # X通道显示
        ax6 = fig.add_subplot(gs1[1, 1])
        x_channel = converted_rgbcx[:, :, 4]
        im3 = ax6.imshow(x_channel, cmap='plasma')
        ax6.set_title("X Channel\n(Extra)", fontsize=12, fontweight='bold')
        ax6.axis('off')
        plt.colorbar(im3, ax=ax6, shrink=0.8)
        
        # 色差分布图
        ax7 = fig.add_subplot(gs1[1, 2])
        delta_e_reshaped = quality_metrics['delta_e_values'].reshape(original_rgbv.shape[:2])
        im4 = ax7.imshow(delta_e_reshaped, cmap='hot_r')
        ax7.set_title("ΔE Distribution\n(Lower is better)", fontsize=12, fontweight='bold')
        ax7.axis('off')
        plt.colorbar(im4, ax=ax7, shrink=0.8)
        
        # RGB差异图
        ax8 = fig.add_subplot(gs1[1, 3])
        rgb_diff = np.mean(np.abs(rgb_display - rgb_converted), axis=2)
        im5 = ax8.imshow(rgb_diff, cmap='hot')
        ax8.set_title("RGB Difference\n(PSO Optimized)", fontsize=12, fontweight='bold')
        ax8.axis('off')
        plt.colorbar(im5, ax=ax8, shrink=0.8)
    
    def _create_gamut_mapping_section(self, fig, converter):
        """创建色域映射图"""
        gs1 = fig.add_gridspec(3, 4, left=0.05, right=0.70, top=0.95, bottom=0.55)
        ax9 = fig.add_subplot(gs1[2, :])
        
        # 绘制色域边界
        rgbv_polygon = Polygon(converter.rgbv_vertices, fill=False, 
                              edgecolor='blue', linewidth=3, label='4-Channel RGBV')
        ax9.add_patch(rgbv_polygon)
        
        rgbcx_polygon = Polygon(converter.rgbcx_vertices, fill=False, 
                               edgecolor='red', linewidth=3, label='5-Channel RGBCX')  
        ax9.add_patch(rgbcx_polygon)
        
        ax9.set_xlim(0, 0.8)
        ax9.set_ylim(0, 0.9)
        ax9.set_xlabel('CIE x')
        ax9.set_ylabel('CIE y')
        ax9.set_title('PSO-Optimized Gamut Mapping')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        ax9.set_aspect('equal')
    
    def _create_statistics_section(self, fig, quality_metrics):
        """创建统计分析区域"""
        gs2 = fig.add_gridspec(3, 2, left=0.75, right=0.98, top=0.95, bottom=0.55)
        
        # 质量分布饼图
        self._create_quality_pie_chart(fig, gs2, quality_metrics)
        
        # ΔE分布直方图
        self._create_delta_e_histogram(fig, gs2, quality_metrics)
        
        # 统计表格
        self._create_statistics_table(fig, gs2, quality_metrics)
    
    def _create_quality_pie_chart(self, fig, gs2, quality_metrics):
        """创建质量分布饼图"""
        ax10 = fig.add_subplot(gs2[0, 0])
        quality_labels = ['Excellent\n(ΔE<1)', 'Good\n(1≤ΔE<3)', 'Acceptable\n(3≤ΔE<6)', 'Poor\n(ΔE≥6)']
        quality_values = [
            quality_metrics['excellent_rate'],
            quality_metrics['good_rate'],
            quality_metrics['acceptable_rate'], 
            quality_metrics['poor_rate']
        ]
        colors = [self.colors['excellent'], self.colors['good'], 
                 self.colors['acceptable'], self.colors['poor']]
        
        wedges, texts, autotexts = ax10.pie(quality_values, labels=quality_labels, colors=colors,
                                           autopct='%1.1f%%', startangle=90)
        ax10.set_title('Quality Distribution\n(PSO Optimized)')
    
    def _create_delta_e_histogram(self, fig, gs2, quality_metrics):
        """创建ΔE分布直方图"""
        ax11 = fig.add_subplot(gs2[0, 1])
        delta_e_values = quality_metrics['delta_e_values']
        ax11.hist(delta_e_values, bins=40, alpha=0.7, color='skyblue', edgecolor='black')
        ax11.axvline(quality_metrics['avg_delta_e'], color='red', linestyle='--',
                    label=f"Mean: {quality_metrics['avg_delta_e']:.3f}")
        ax11.axvline(quality_metrics['median_delta_e'], color='orange', linestyle='--',
                    label=f"Median: {quality_metrics['median_delta_e']:.3f}")
        ax11.set_xlabel('ΔE*ab')
        ax11.set_ylabel('Frequency')
        ax11.set_title('ΔE Distribution')
        ax11.legend()
        ax11.grid(True, alpha=0.3)
    
    def _create_statistics_table(self, fig, gs2, quality_metrics):
        """创建统计表格"""
        ax12 = fig.add_subplot(gs2[1:, :])
        ax12.axis('off')
        
        stats_data = [
            ['Metric', 'Value', 'Improvement'],
            ['Average ΔE*ab', f'{quality_metrics["avg_delta_e"]:.4f}', 'PSO Optimized'],
            ['Median ΔE*ab', f'{quality_metrics["median_delta_e"]:.4f}', 'Lower is Better'],
            ['Max ΔE*ab', f'{quality_metrics["max_delta_e"]:.4f}', 'Reduced Range'],
            ['Std ΔE*ab', f'{quality_metrics["std_delta_e"]:.4f}', 'More Consistent'],
            ['Excellent Rate', f'{quality_metrics["excellent_rate"]:.1f}%', 'ΔE < 1.0'],
            ['Good Rate', f'{quality_metrics["good_rate"]:.1f}%', '1.0 ≤ ΔE < 3.0'],
            ['Acceptable Rate', f'{quality_metrics["acceptable_rate"]:.1f}%', '3.0 ≤ ΔE < 6.0'],
            ['Poor Rate', f'{quality_metrics["poor_rate"]:.1f}%', 'ΔE ≥ 6.0'],
        ]
        
        table = ax12.table(cellText=stats_data, cellLoc='center', loc='center',
                          colWidths=[0.35, 0.25, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 表格样式设置
        for i in range(len(stats_data)):
            if i == 0:
                for j in range(3):
                    table[(i, j)].set_facecolor('#4CAF50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, 0)].set_facecolor('#f0f0f0')
                table[(i, 1)].set_facecolor('#ffffff')
                table[(i, 2)].set_facecolor('#f8f8f8')
    
    def _create_algorithm_explanation_section(self, fig):
        """创建算法说明区域"""
        gs3 = fig.add_gridspec(1, 1, left=0.05, right=0.98, top=0.45, bottom=0.05)
        ax13 = fig.add_subplot(gs3[0, 0])
        ax13.axis('off')
        
        explanation_text = """
    PSO-OPTIMIZED 4-Channel to 5-Channel Color Conversion:
    
    Algorithm Features:
    • Particle Swarm Optimization (PSO) for transformation matrix optimization
    • CIE ΔE*ab objective function for perceptual color difference minimization
    • 29-parameter optimization space (20 matrix + 9 channel parameters)
    • Multi-population convergence with adaptive learning factors
    
    PSO Parameters:
    • Swarm Size: 30 particles
    • Iterations: 100 generations  
    • Inertia Weight: 0.7
    • Learning Factors: c1=1.5, c2=1.5
    
    Optimization Strategy:
    • Global search for optimal transformation matrix
    • Balanced exploration vs exploitation
    • Regularization to prevent overfitting
    • Boundary constraints for numerical stability
    
    Key Improvements:
    • Data-driven parameter optimization
    • Reduced color difference through intelligent mapping
    • Enhanced V-channel decomposition strategy
    • Professional-grade conversion quality
        """
        
        ax13.text(0.02, 0.98, explanation_text, transform=ax13.transAxes, fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    def create_convergence_plot(self, convergence_history, save_dir):
        """创建PSO收敛曲线图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        iterations = range(len(convergence_history))
        ax.plot(iterations, convergence_history, 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('PSO Iteration')
        ax.set_ylabel('Best ΔE*ab Score')
        ax.set_title('PSO Optimization Convergence', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加收敛标注
        final_score = convergence_history[-1]
        ax.annotate(f'Final Score: {final_score:.4f}', 
                   xy=(len(convergence_history)-1, final_score),
                   xytext=(len(convergence_history)*0.7, final_score*1.2),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'pso_convergence.png'), dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def create_transformation_matrix_heatmap(self, transformation_matrix, save_dir):
        """创建转换矩阵热力图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(transformation_matrix, cmap='RdBu', aspect='auto')
        
        # 设置坐标轴
        ax.set_xticks(range(5))
        ax.set_xticklabels(['R', 'G', 'B', 'C', 'X'])
        ax.set_yticks(range(4))
        ax.set_yticklabels(['R', 'G', 'B', 'V'])
        ax.set_xlabel('Output Channels (RGBCX)')
        ax.set_ylabel('Input Channels (RGBV)')
        ax.set_title('PSO-Optimized Transformation Matrix', fontsize=14, fontweight='bold')
        
        # 添加数值标注
        for i in range(4):
            for j in range(5):
                text = ax.text(j, i, f'{transformation_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Transformation Coefficient')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'transformation_matrix_heatmap.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def create_channel_comparison(self, original_rgbv, converted_rgbcx, save_dir):
        """创建通道对比图"""
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        
        # 原始RGBV通道
        channels_orig = ['R', 'G', 'B', 'V']
        for i in range(4):
            axes[0, i].imshow(original_rgbv[:, :, i], cmap='viridis')
            axes[0, i].set_title(f'Original {channels_orig[i]}', fontweight='bold')
            axes[0, i].axis('off')
        
        # 空白第5个子图
        axes[0, 4].axis('off')
        
        # 转换后RGBCX通道
        channels_conv = ['R', 'G', 'B', 'C', 'X']
        for i in range(5):
            axes[1, i].imshow(converted_rgbcx[:, :, i], cmap='viridis')
            axes[1, i].set_title(f'Converted {channels_conv[i]}', fontweight='bold')
            axes[1, i].axis('off')
        
        fig.suptitle('Channel-by-Channel Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'channel_comparison.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def create_quality_analysis_dashboard(self, quality_metrics, save_dir):
        """创建质量分析仪表板"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 质量分布饼图
        quality_labels = ['Excellent', 'Good', 'Acceptable', 'Poor']
        quality_values = [
            quality_metrics['excellent_rate'],
            quality_metrics['good_rate'],
            quality_metrics['acceptable_rate'], 
            quality_metrics['poor_rate']
        ]
        colors = [self.colors['excellent'], self.colors['good'], 
                 self.colors['acceptable'], self.colors['poor']]
        
        ax1.pie(quality_values, labels=quality_labels, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Quality Distribution')
        
        # 2. ΔE直方图
        delta_e_values = quality_metrics['delta_e_values']
        ax2.hist(delta_e_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(quality_metrics['avg_delta_e'], color='red', linestyle='--', 
                   label=f"Mean: {quality_metrics['avg_delta_e']:.3f}")
        ax2.set_xlabel('ΔE*ab')
        ax2.set_ylabel('Frequency')
        ax2.set_title('ΔE Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 累积分布函数
        sorted_delta_e = np.sort(delta_e_values)
        cumulative = np.arange(1, len(sorted_delta_e) + 1) / len(sorted_delta_e) * 100
        ax3.plot(sorted_delta_e, cumulative, 'b-', linewidth=2)
        ax3.axvline(1.0, color='red', linestyle='--', alpha=0.7, label='ΔE=1.0')
        ax3.axvline(3.0, color='orange', linestyle='--', alpha=0.7, label='ΔE=3.0')
        ax3.axvline(6.0, color='red', linestyle=':', alpha=0.7, label='ΔE=6.0')
        ax3.set_xlabel('ΔE*ab')
        ax3.set_ylabel('Cumulative Percentage (%)')
        ax3.set_title('Cumulative Distribution Function')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 统计指标条形图
        metrics = ['Mean', 'Median', 'Std', 'Max']
        values = [
            quality_metrics['avg_delta_e'],
            quality_metrics['median_delta_e'],
            quality_metrics['std_delta_e'],
            quality_metrics['max_delta_e']
        ]
        bars = ax4.bar(metrics, values, color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])
        ax4.set_ylabel('ΔE*ab')
        ax4.set_title('Statistical Metrics')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'quality_analysis_dashboard.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()

# 使用示例函数
def visualize_pso_results(original_rgbv, converted_rgbcx, converter, quality_metrics, 
                         transformation_matrix, save_dir):
    """PSO结果可视化的主函数"""
    visualizer = PSO_ColorConversionVisualizer()
    
    # 创建主要的综合分析图
    visualizer.create_comprehensive_visualization(
        original_rgbv, converted_rgbcx, converter, 
        quality_metrics, transformation_matrix, save_dir
    )
    
    # 创建额外的分析图表
    visualizer.create_transformation_matrix_heatmap(transformation_matrix, save_dir)
    visualizer.create_channel_comparison(original_rgbv, converted_rgbcx, save_dir)
    visualizer.create_quality_analysis_dashboard(quality_metrics, save_dir)
    
    print("所有可视化图表已生成完成！")

if __name__ == "__main__":
    print("PSO颜色转换可视化模块已加载")
    print("主要功能：")
    print("1. 综合分析图 - create_comprehensive_visualization()")
    print("2. 收敛曲线图 - create_convergence_plot()")
    print("3. 转换矩阵热力图 - create_transformation_matrix_heatmap()")
    print("4. 通道对比图 - create_channel_comparison()")
    print("5. 质量分析仪表板 - create_quality_analysis_dashboard()")
