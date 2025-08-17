#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2改进版数学模型可视化
生成模型流程图和关键公式图表
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_model_flowchart():
    """创建数学模型流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # 定义颜色
    colors = {
        'input': '#E8F4FD',
        'process': '#B3E5FC', 
        'model': '#81C784',
        'output': '#FFB74D',
        'decision': '#F8BBD9'
    }
    
    # 定义框的位置和大小
    boxes = {
        'data': (2, 10, 3, 1.2),
        'preprocess': (2, 8, 3, 1.2),
        'indicators': (2, 6, 3, 1.2),
        'risk_score': (7, 8, 3, 1.2),
        'logistic': (7, 6, 3, 1.2),
        'industry': (12, 8, 3, 1.2),
        'credit_limit': (7, 4, 3, 1.2),
        'interest_rate': (12, 6, 3, 1.2),
        'optimization': (7, 2, 3, 1.2),
        'result': (12, 2, 3, 1.2)
    }
    
    # 绘制框和文本
    box_info = {
        'data': ('Invoice Data\n(Sales/Purchase)', colors['input']),
        'preprocess': ('Data Preprocessing\n& Feature Engineering', colors['process']),
        'indicators': ('Financial Indicators\nCalculation', colors['process']),
        'risk_score': ('5-Dimension\nRisk Scoring', colors['model']),
        'logistic': ('Logistic Transform\nP(default)', colors['model']),
        'industry': ('Industry Sensitivity\nParameters', colors['model']),
        'credit_limit': ('Credit Limit\nCalculation', colors['process']),
        'interest_rate': ('Dynamic Interest\nRate Pricing', colors['process']),
        'optimization': ('Risk-Adjusted\nReturn Optimization', colors['decision']),
        'result': ('Credit Strategy\nResults', colors['output'])
    }
    
    for box_name, (x, y, w, h) in boxes.items():
        text, color = box_info[box_name]
        
        # 绘制圆角矩形
        fancy_box = FancyBboxPatch((x, y), w, h,
                                   boxstyle="round,pad=0.1",
                                   facecolor=color,
                                   edgecolor='black',
                                   linewidth=1.5)
        ax.add_patch(fancy_box)
        
        # 添加文本
        ax.text(x + w/2, y + h/2, text, 
                ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # 绘制箭头连接
    connections = [
        ('data', 'preprocess'),
        ('preprocess', 'indicators'), 
        ('indicators', 'risk_score'),
        ('risk_score', 'logistic'),
        ('risk_score', 'credit_limit'),
        ('logistic', 'credit_limit'),
        ('industry', 'risk_score'),
        ('industry', 'interest_rate'),
        ('logistic', 'interest_rate'),
        ('credit_limit', 'optimization'),
        ('interest_rate', 'optimization'),
        ('optimization', 'result')
    ]
    
    for start, end in connections:
        x1, y1, w1, h1 = boxes[start]
        x2, y2, w2, h2 = boxes[end]
        
        # 计算连接点
        if x1 < x2:  # 向右连接
            start_point = (x1 + w1, y1 + h1/2)
            end_point = (x2, y2 + h2/2)
        elif x1 > x2:  # 向左连接
            start_point = (x1, y1 + h1/2)
            end_point = (x2 + w2, y2 + h2/2)
        else:  # 向下连接
            start_point = (x1 + w1/2, y1)
            end_point = (x2 + w2/2, y2 + h2)
        
        # 绘制箭头
        ax.annotate('', xy=end_point, xytext=start_point,
                   arrowprops=dict(arrowstyle='->', 
                                 connectionstyle='arc3,rad=0.1',
                                 color='darkblue', lw=2))
    
    # 添加关键公式
    formulas = [
        (1, 4, r'$R_i = \sum_{j=1}^5 w_j \cdot S_{j,i}$', 'Risk Score'),
        (1, 3, r'$P_i = \frac{1}{1+e^{0.483+3.076R_i}}$', 'Default Probability'),
        (1, 2, r'$L_i = P_i \cdot \gamma_j \cdot 0.6$', 'Expected Loss'),
        (1, 1, r'$\max \sum C_i \cdot (r_i - L_i) \cdot R_i$', 'Objective Function')
    ]
    
    for x, y, formula, label in formulas:
        ax.text(x, y, f'{label}:\n{formula}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'),
                fontsize=9, ha='left')
    
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.title('Problem 2 Improved Mathematical Model Flowchart', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('model_flowchart_improved.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_formula_visualization():
    """创建关键公式可视化"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 风险评分构成
    categories = ['Financial', 'Stability', 'Efficiency', 'Market', 'Industry']
    weights = [0.35, 0.25, 0.20, 0.12, 0.08]
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
    
    ax1.pie(weights, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Risk Score Composition\n$R_i = \\sum_{j=1}^5 w_j \\cdot S_{j,i}$', 
                  fontsize=12, fontweight='bold')
    
    # 2. Logistic转换曲线
    risk_scores = np.linspace(0, 1, 100)
    alpha, beta = 0.483, 3.076
    default_probs = 1 / (1 + np.exp(alpha + beta * risk_scores))
    
    ax2.plot(risk_scores, default_probs, 'b-', linewidth=3, label='Logistic Curve')
    ax2.scatter([0.2, 0.8], [0.25, 0.05], color='red', s=100, zorder=5, 
               label='Calibration Points')
    ax2.set_xlabel('Risk Score $R_i$')
    ax2.set_ylabel('Default Probability $P_i$')
    ax2.set_title('Logistic Transform: Risk Score → Default Probability\n' + 
                  '$P_i = \\frac{1}{1+e^{0.483+3.076R_i}}$', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. 行业敏感性参数
    industries = ['Manufacturing', 'Wholesale', 'Service', 'Construction', 'Others']
    risk_coeffs = [0.8, 1.0, 0.9, 1.3, 1.1]
    sensitivity = [1.2, 1.5, 1.1, 1.8, 1.3]
    
    x = np.arange(len(industries))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, risk_coeffs, width, label='Risk Coefficient $\\alpha_j$', 
                    color='lightblue')
    bars2 = ax3.bar(x + width/2, sensitivity, width, label='Sensitivity $\\gamma_j$', 
                    color='lightcoral')
    
    ax3.set_xlabel('Industry')
    ax3.set_ylabel('Parameter Value')
    ax3.set_title('Industry Sensitivity Parameters', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(industries, rotation=45)
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)
    
    # 4. 目标函数对比
    models = ['Original', 'Improved']
    formulas = ['$\\max \\sum L_i \\cdot r_i \\cdot R_i$', 
               '$\\max \\sum L_i \\cdot (r_i - L_i) \\cdot R_i$']
    issues = ['Logic Contradiction', 'Risk-Return Balance']
    
    colors_comp = ['lightcoral', 'lightgreen']
    
    for i, (model, formula, issue, color) in enumerate(zip(models, formulas, issues, colors_comp)):
        rect = patches.Rectangle((i*0.4, 0), 0.35, 1, linewidth=2, 
                               edgecolor='black', facecolor=color)
        ax4.add_patch(rect)
        
        ax4.text(i*0.4 + 0.175, 0.7, model, ha='center', va='center', 
                fontweight='bold', fontsize=12)
        ax4.text(i*0.4 + 0.175, 0.5, formula, ha='center', va='center', 
                fontsize=10)
        ax4.text(i*0.4 + 0.175, 0.3, issue, ha='center', va='center', 
                fontsize=9, style='italic')
    
    ax4.set_xlim(-0.05, 0.85)
    ax4.set_ylim(0, 1)
    ax4.set_title('Objective Function Comparison', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('formula_visualization_improved.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_risk_distribution_analysis():
    """创建风险分布分析图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 模拟改进版数据
    np.random.seed(42)
    n_enterprises = 302
    
    # 1. 违约概率分布
    # 基于Logistic模型生成
    risk_scores = np.random.beta(2, 3, n_enterprises)  # 偏向较高风险评分
    alpha, beta = 0.483, 3.076
    default_probs = 1 / (1 + np.exp(alpha + beta * risk_scores))
    
    ax1.hist(default_probs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(default_probs.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {default_probs.mean():.3f}')
    ax1.set_xlabel('Default Probability')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Default Probability Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 行业分布
    industries = ['Manufacturing', 'Wholesale', 'Service', 'Construction', 'Others']
    industry_counts = [96, 71, 59, 47, 29]
    colors_ind = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
    
    wedges, texts, autotexts = ax2.pie(industry_counts, labels=industries, 
                                      colors=colors_ind, autopct='%1.1f%%', 
                                      startangle=90)
    ax2.set_title('Industry Distribution\n(302 Enterprises)', 
                  fontsize=12, fontweight='bold')
    
    # 3. 风险评分vs违约概率散点图
    ax3.scatter(risk_scores, default_probs, alpha=0.6, c=default_probs, 
               cmap='coolwarm', s=30)
    
    # 拟合曲线
    x_fit = np.linspace(0, 1, 100)
    y_fit = 1 / (1 + np.exp(alpha + beta * x_fit))
    ax3.plot(x_fit, y_fit, 'r-', linewidth=3, label='Logistic Curve')
    
    ax3.set_xlabel('Risk Score $R_i$')
    ax3.set_ylabel('Default Probability $P_i$')
    ax3.set_title('Risk Score vs Default Probability\n' + 
                  '$P_i = \\frac{1}{1+e^{0.483+3.076R_i}}$', 
                  fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 风险调整收益分布
    # 模拟利率和期望损失率
    interest_rates = 0.04 + (1 - risk_scores) * 0.11
    expected_losses = default_probs * 1.3 * 0.6  # 平均敏感性系数
    risk_adjusted_returns = interest_rates - expected_losses
    
    ax4.hist(risk_adjusted_returns, bins=25, alpha=0.7, color='lightgreen', 
            edgecolor='black')
    ax4.axvline(risk_adjusted_returns.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {risk_adjusted_returns.mean():.3f}')
    ax4.set_xlabel('Risk-Adjusted Return Rate')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Risk-Adjusted Return Distribution\n$RAR_i = r_i - L_i$', 
                  fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('risk_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    print("🎨 生成问题2改进版数学模型可视化...")
    
    # 创建模型流程图
    create_model_flowchart()
    print("   ✅ 模型流程图已生成: model_flowchart_improved.png")
    
    # 创建公式可视化
    create_formula_visualization()
    print("   ✅ 公式可视化已生成: formula_visualization_improved.png")
    
    # 创建风险分布分析
    create_risk_distribution_analysis()
    print("   ✅ 风险分布分析已生成: risk_distribution_analysis.png")
    
    print("\n🎉 所有数学模型可视化图表生成完成!")

if __name__ == "__main__":
    main()
