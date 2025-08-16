#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
银行信贷策略数学模型公式可视化
展示完整的数学模型公式和推导过程
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_mathematical_model_diagram():
    """创建数学模型图表"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 20))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 24)
    ax.axis('off')
    
    # 标题
    ax.text(5, 23, '银行信贷策略数学模型', fontsize=24, fontweight='bold', ha='center')
    ax.text(5, 22.3, 'Mathematical Model for Bank Credit Strategy', fontsize=14, ha='center', style='italic')
    
    # 1. 约束条件
    box1 = FancyBboxPatch((0.5, 20.5), 9, 1.5, boxstyle="round,pad=0.1", 
                          facecolor='lightblue', edgecolor='navy', linewidth=2)
    ax.add_patch(box1)
    ax.text(5, 21.7, '1. 约束条件 (Constraints)', fontsize=16, fontweight='bold', ha='center')
    ax.text(5, 21.3, '贷款额度: 10万元 ≤ L ≤ 100万元', fontsize=12, ha='center')
    ax.text(5, 21.0, '年利率: 4% ≤ r ≤ 15%', fontsize=12, ha='center')
    ax.text(5, 20.7, '贷款期限: T = 1年', fontsize=12, ha='center')
    
    # 2. 综合风险评分模型
    box2 = FancyBboxPatch((0.5, 18.0), 9, 2.2, boxstyle="round,pad=0.1", 
                          facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(box2)
    ax.text(5, 19.9, '2. 综合风险评分模型 (Risk Scoring Model)', fontsize=16, fontweight='bold', ha='center')
    ax.text(5, 19.5, r'$R_i = 0.35 \cdot R_{C_i} + 0.25 \cdot D_i + 0.20 \cdot R_{F_i} + 0.15 \cdot R_{Q_i} + 0.05 \cdot R_{S_i}$', 
            fontsize=14, ha='center')
    ax.text(1, 19.1, '• 信誉评级风险: $R_{C_i} ∈ \\{0.1, 0.3, 0.6, 0.9\\}$ 对应 {A,B,C,D}', fontsize=10)
    ax.text(1, 18.8, '• 历史违约风险: $R_{D_i} = 0.8 \\cdot D_i + 0.1 \\cdot (1-D_i)$', fontsize=10)
    ax.text(1, 18.5, '• 财务状况风险: $R_{F_i} = 1 - \\frac{毛利率_i - \\min(毛利率)}{\\max(毛利率) - \\min(毛利率)}$', fontsize=10)
    ax.text(1, 18.2, '• 发票质量风险: $R_{Q_i} = \\frac{作废发票数量_i}{总发票数量_i}$', fontsize=10)
    
    # 3. 贷款额度决策模型
    box3 = FancyBboxPatch((0.5, 15.0), 9, 2.7, boxstyle="round,pad=0.1", 
                          facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(box3)
    ax.text(5, 17.4, '3. 贷款额度决策模型 (Loan Amount Decision Model)', fontsize=16, fontweight='bold', ha='center')
    
    # 基础额度公式
    ax.text(1, 17.0, '基础额度计算:', fontsize=12, fontweight='bold')
    ax.text(1, 16.7, '• A级无违约: $L_{base} = \\min(100, \\max(10, 0.25 \\times 年营业收入))$', fontsize=10)
    ax.text(1, 16.4, '• A级有违约: $L_{base} = \\min(80, \\max(10, 0.15 \\times 年营业收入))$', fontsize=10)
    ax.text(1, 16.1, '• B级无违约: $L_{base} = \\min(80, \\max(10, 0.20 \\times 年营业收入))$', fontsize=10)
    ax.text(1, 15.8, '• C级无违约: $L_{base} = \\min(50, \\max(10, 0.10 \\times 年营业收入))$', fontsize=10)
    ax.text(1, 15.5, '• D级企业: $L_{base} = 0$ (拒贷)', fontsize=10)
    ax.text(1, 15.2, '最终约束: $L_i = \\min(100, \\max(10, L_{adj,i}))$ 如果 $L_{adj,i} ≥ 10$，否则 $L_i = 0$', fontsize=10)
    
    # 4. 利率定价模型
    box4 = FancyBboxPatch((0.5, 12.2), 9, 2.5, boxstyle="round,pad=0.1", 
                          facecolor='lightcoral', edgecolor='darkred', linewidth=2)
    ax.add_patch(box4)
    ax.text(5, 14.4, '4. 利率定价模型 (Interest Rate Pricing Model)', fontsize=16, fontweight='bold', ha='center')
    
    ax.text(1, 14.0, '基础利率计算:', fontsize=12, fontweight='bold')
    ax.text(1, 13.7, '• A级无违约: $r_{base} = 0.045 + 0.02 \\times R_i$', fontsize=10)
    ax.text(1, 13.4, '• A级有违约: $r_{base} = 0.055 + 0.03 \\times R_i$', fontsize=10)
    ax.text(1, 13.1, '• B级无违约: $r_{base} = 0.055 + 0.03 \\times R_i$', fontsize=10)
    ax.text(1, 12.8, '• C级无违约: $r_{base} = 0.080 + 0.05 \\times R_i$', fontsize=10)
    ax.text(1, 12.5, '利率约束: $r_i = \\max(0.04, \\min(0.15, r_{base,i}))$', fontsize=12, fontweight='bold')
    
    # 5. 违约概率与收益模型
    box5 = FancyBboxPatch((0.5, 9.0), 9, 2.9, boxstyle="round,pad=0.1", 
                          facecolor='lightpink', edgecolor='purple', linewidth=2)
    ax.add_patch(box5)
    ax.text(5, 11.6, '5. 违约概率与收益模型 (Default Probability & Revenue Model)', fontsize=16, fontweight='bold', ha='center')
    
    ax.text(1, 11.2, '违约概率模型:', fontsize=12, fontweight='bold')
    ax.text(1, 10.9, '$P_i = \\min(0.95, R_i \\times 1.1)$', fontsize=12)
    
    ax.text(1, 10.5, '预期收益计算:', fontsize=12, fontweight='bold')
    ax.text(1, 10.2, '$E_i = L_i \\times r_i \\times (1 - P_i)$', fontsize=12)
    
    ax.text(1, 9.8, '风险调整收益:', fontsize=12, fontweight='bold')
    ax.text(1, 9.5, '$RAR_i = \\frac{E_i}{L_i \\times R_i + 0.01}$', fontsize=12)
    
    ax.text(1, 9.2, '年利息收入: $I_i = L_i \\times r_i$', fontsize=10)
    
    # 6. 优化目标函数
    box6 = FancyBboxPatch((0.5, 6.0), 9, 2.7, boxstyle="round,pad=0.1", 
                          facecolor='lightsteelblue', edgecolor='navy', linewidth=2)
    ax.add_patch(box6)
    ax.text(5, 8.4, '6. 优化目标函数 (Optimization Objective Function)', fontsize=16, fontweight='bold', ha='center')
    
    ax.text(2.5, 7.9, '最大化总预期收益:', fontsize=14, fontweight='bold')
    ax.text(5, 7.5, r'$\max \sum_{i=1}^{n} E_i = \sum_{i=1}^{n} L_i \times r_i \times (1 - P_i)$', 
            fontsize=16, ha='center')
    
    ax.text(1, 7.0, '约束条件:', fontsize=12, fontweight='bold')
    ax.text(1, 6.7, '• 预算约束: $\\sum_{i=1}^{n} L_i ≤ B$', fontsize=11)
    ax.text(1, 6.4, '• 额度约束: $10 ≤ L_i ≤ 100$ (对于获贷企业)', fontsize=11)
    ax.text(1, 6.1, '• 利率约束: $0.04 ≤ r_i ≤ 0.15$', fontsize=11)
    
    # 7. 求解算法
    box7 = FancyBboxPatch((0.5, 3.0), 9, 2.7, boxstyle="round,pad=0.1", 
                          facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(box7)
    ax.text(5, 5.4, '7. 预算分配优化算法 (Budget Allocation Algorithm)', fontsize=16, fontweight='bold', ha='center')
    
    ax.text(1, 5.0, '步骤1: 计算每个企业的风险调整收益 $RAR_i$', fontsize=11)
    ax.text(1, 4.7, '步骤2: 按 $RAR_i$ 降序排列所有企业', fontsize=11)
    ax.text(1, 4.4, '步骤3: 依次分配贷款额度，直到预算耗尽', fontsize=11)
    ax.text(1, 4.1, '步骤4: 对最后一家企业分配剩余预算(如果≥10万元)', fontsize=11)
    ax.text(1, 3.8, '步骤5: 重新计算所有收益指标', fontsize=11)
    
    ax.text(2, 3.4, '排序准则: $RAR_1 ≥ RAR_2 ≥ ... ≥ RAR_n$', fontsize=12, style='italic')
    
    # 8. 模型输出
    box8 = FancyBboxPatch((0.5, 0.2), 9, 2.5, boxstyle="round,pad=0.1", 
                          facecolor='lightcyan', edgecolor='teal', linewidth=2)
    ax.add_patch(box8)
    ax.text(5, 2.4, '8. 模型输出 (Model Output)', fontsize=16, fontweight='bold', ha='center')
    
    ax.text(1, 2.0, '• 获贷企业数量及名单', fontsize=11)
    ax.text(1, 1.7, '• 每家企业的贷款额度 $L_i$ 和利率 $r_i$', fontsize=11)
    ax.text(1, 1.4, '• 预期总收益: $\\sum E_i$，投资回报率: $\\frac{\\sum E_i}{\\sum L_i}$', fontsize=11)
    ax.text(1, 1.1, '• 风险控制指标: 平均违约概率、风险分布', fontsize=11)
    ax.text(1, 0.8, '• 约束满足情况验证', fontsize=11)
    ax.text(1, 0.5, '• 资源配置效率分析', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('数学模型公式图.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 数学模型公式图已生成: 数学模型公式图.png")

def create_model_flowchart():
    """创建模型流程图"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 18))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # 标题
    ax.text(5, 19.5, '银行信贷策略模型流程图', fontsize=20, fontweight='bold', ha='center')
    ax.text(5, 19, 'Credit Strategy Model Flowchart', fontsize=14, ha='center', style='italic')
    
    # 流程框
    boxes = [
        {'pos': (5, 17.5), 'size': (3.5, 0.8), 'text': '数据输入\n企业信息、发票数据', 'color': 'lightblue'},
        {'pos': (5, 16), 'size': (3.5, 0.8), 'text': '财务指标计算\n营业收入、毛利率等', 'color': 'lightgreen'},
        {'pos': (5, 14.5), 'size': (3.5, 0.8), 'text': '风险评分计算\nR_i = Σw_j × R_j', 'color': 'lightcoral'},
        {'pos': (2, 13), 'size': (3, 0.8), 'text': '贷款额度决策\nL_i计算', 'color': 'lightyellow'},
        {'pos': (8, 13), 'size': (3, 0.8), 'text': '利率定价\nr_i计算', 'color': 'lightpink'},
        {'pos': (5, 11.5), 'size': (3.5, 0.8), 'text': '违约概率估算\nP_i = min(0.95, R_i×1.1)', 'color': 'lightgray'},
        {'pos': (5, 10), 'size': (3.5, 0.8), 'text': '预期收益计算\nE_i = L_i × r_i × (1-P_i)', 'color': 'lightsteelblue'},
        {'pos': (5, 8.5), 'size': (3.5, 0.8), 'text': '风险调整收益\nRAR_i = E_i/(L_i×R_i)', 'color': 'lightcyan'},
        {'pos': (5, 7), 'size': (3.5, 0.8), 'text': '按RAR_i排序\n优化资源配置', 'color': 'lavender'},
        {'pos': (2, 5.5), 'size': (2.8, 0.8), 'text': '预算约束\n检查', 'color': 'wheat'},
        {'pos': (8, 5.5), 'size': (2.8, 0.8), 'text': '约束条件\n验证', 'color': 'mistyrose'},
        {'pos': (5, 4), 'size': (3.5, 0.8), 'text': '最终策略输出\n获贷名单及条件', 'color': 'lightblue'},
        {'pos': (5, 2.5), 'size': (3.5, 0.8), 'text': '结果分析与报告\n收益、风险指标', 'color': 'lightgreen'},
    ]
    
    # 绘制流程框
    for box in boxes:
        rect = FancyBboxPatch(
            (box['pos'][0] - box['size'][0]/2, box['pos'][1] - box['size'][1]/2),
            box['size'][0], box['size'][1],
            boxstyle="round,pad=0.1", 
            facecolor=box['color'], 
            edgecolor='black', 
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(box['pos'][0], box['pos'][1], box['text'], 
               ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 绘制箭头
    arrows = [
        ((5, 17.1), (5, 16.4)),    # 1->2
        ((5, 15.6), (5, 14.9)),    # 2->3
        ((4.2, 14.1), (2.8, 13.4)),  # 3->4
        ((5.8, 14.1), (7.2, 13.4)),  # 3->5
        ((2, 12.6), (4.2, 11.9)),    # 4->6
        ((8, 12.6), (5.8, 11.9)),    # 5->6
        ((5, 11.1), (5, 10.4)),    # 6->7
        ((5, 9.6), (5, 8.9)),      # 7->8
        ((5, 8.1), (5, 7.4)),      # 8->9
        ((4.2, 6.6), (2.8, 5.9)),     # 9->10
        ((5.8, 6.6), (7.2, 5.9)),     # 9->11
        ((2.8, 5.1), (4.2, 4.4)),     # 10->12
        ((7.2, 5.1), (5.8, 4.4)),     # 11->12
        ((5, 3.6), (5, 2.9)),      # 12->13
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='navy'))
    
    # 添加判断框
    decision_box = patches.RegularPolygon((5, 5.5), 4, radius=0.6, 
                                        orientation=np.pi/4, 
                                        facecolor='yellow', 
                                        edgecolor='black', 
                                        linewidth=2)
    ax.add_patch(decision_box)
    ax.text(5, 5.5, '约束\n检查', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 连接判断框的箭头
    ax.annotate('', xy=(4.4, 5.5), xytext=(2.8, 5.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='navy'))
    ax.annotate('', xy=(7.2, 5.5), xytext=(5.6, 5.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='navy'))
    ax.annotate('', xy=(5, 4.4), xytext=(5, 4.9),
               arrowprops=dict(arrowstyle='->', lw=2, color='navy'))
    
    # 添加说明文字
    ax.text(0.5, 1, '说明: 该流程图展示了银行信贷策略数学模型的完整计算流程，\n从数据输入到最终策略输出的每个关键步骤。', 
           fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('模型流程图.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 模型流程图已生成: 模型流程图.png")

def main():
    """主函数"""
    print("📊 生成银行信贷策略数学模型图表...")
    
    # 生成数学公式图
    create_mathematical_model_diagram()
    
    # 生成流程图
    create_model_flowchart()
    
    print("\n🎉 数学模型图表生成完成!")
    print("📁 生成的文件:")
    print("   📈 数学模型公式图.png")
    print("   📊 模型流程图.png")

if __name__ == "__main__":
    main()
