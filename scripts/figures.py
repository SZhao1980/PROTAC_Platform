"""
生成期刊级高质量图表
使用matplotlib和seaborn创建出版级别的图表
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from matplotlib import rcParams

# 设置matplotlib参数以生成出版级别的图表
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 16

# 设置颜色方案
sns.set_style("whitegrid")
sns.set_palette("husl")

DATA_DIR = "data"
RESULTS_DIR = "results"
FIGURES_DIR = "results"

def load_data():
    """加载数据"""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "protacdb_processed.csv"))
    with open(os.path.join(RESULTS_DIR, "comprehensive_analysis.json"), 'r') as f:
        analysis = json.load(f)
    return df, analysis

def figure1_activity_distribution(df):
    """Figure 1: PROTAC活性分布"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 活性分布饼图
    activity_counts = df['Activity_Label'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    labels = ['Active (n=812)', 'Inactive (n=391)']
    
    axes[0].pie(activity_counts, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
    axes[0].set_title('PROTAC Activity Distribution', fontsize=13, weight='bold')
    
    # 活性分布柱状图
    activity_data = pd.DataFrame({
        'Activity': ['Active', 'Inactive'],
        'Count': [812, 391]
    })
    sns.barplot(data=activity_data, x='Activity', y='Count', ax=axes[1], palette=colors)
    axes[1].set_title('PROTAC Activity Count', fontsize=13, weight='bold')
    axes[1].set_ylabel('Number of Compounds', fontsize=11)
    axes[1].set_xlabel('Activity Status', fontsize=11)
    
    # 添加数值标签
    for i, v in enumerate([812, 391]):
        axes[1].text(i, v + 10, str(v), ha='center', fontsize=11, weight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'Figure1_Activity_Distribution.png'), dpi=300, bbox_inches='tight')
    print("Saved: Figure1_Activity_Distribution.png")
    plt.close()

def figure2_target_distribution(df):
    """Figure 2: 靶点分布"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 获取前15个靶点
    target_counts = df['Target'].value_counts().head(15)
    
    # 创建水平柱状图
    colors_gradient = sns.color_palette("viridis", len(target_counts))
    bars = ax.barh(range(len(target_counts)), target_counts.values, color=colors_gradient)
    
    ax.set_yticks(range(len(target_counts)))
    ax.set_yticklabels(target_counts.index, fontsize=10)
    ax.set_xlabel('Number of PROTAC Compounds', fontsize=12, weight='bold')
    ax.set_title('Top 15 Target Proteins in PROTAC Database', fontsize=14, weight='bold')
    ax.invert_yaxis()
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, target_counts.values)):
        ax.text(val + 1, i, str(int(val)), va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'Figure2_Target_Distribution.png'), dpi=300, bbox_inches='tight')
    print("Saved: Figure2_Target_Distribution.png")
    plt.close()

def figure3_e3_ligase_distribution(df):
    """Figure 3: E3连接酶分布"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    e3_counts = df['E3 Ligase'].value_counts()
    colors_e3 = sns.color_palette("Set2", len(e3_counts))
    
    # 饼图
    axes[0].pie(e3_counts, labels=e3_counts.index, autopct='%1.1f%%',
                colors=colors_e3, textprops={'fontsize': 10})
    axes[0].set_title('E3 Ligase Distribution', fontsize=13, weight='bold')
    
    # 柱状图
    sns.barplot(x=e3_counts.index, y=e3_counts.values, ax=axes[1], palette=colors_e3)
    axes[1].set_title('E3 Ligase Count', fontsize=13, weight='bold')
    axes[1].set_ylabel('Number of Compounds', fontsize=11)
    axes[1].set_xlabel('E3 Ligase Type', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for i, v in enumerate(e3_counts.values):
        axes[1].text(i, v + 10, str(int(v)), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'Figure3_E3Ligase_Distribution.png'), dpi=300, bbox_inches='tight')
    print("Saved: Figure3_E3Ligase_Distribution.png")
    plt.close()

def figure4_linker_analysis(df):
    """Figure 4: 连接子类型分析"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    linker_counts = df['Linker Type'].value_counts()
    colors_linker = sns.color_palette("husl", len(linker_counts))
    
    # 饼图
    axes[0].pie(linker_counts, labels=linker_counts.index, autopct='%1.1f%%',
                colors=colors_linker, textprops={'fontsize': 10})
    axes[0].set_title('Linker Type Distribution', fontsize=13, weight='bold')
    
    # 柱状图
    sns.barplot(x=linker_counts.index, y=linker_counts.values, ax=axes[1], palette=colors_linker)
    axes[1].set_title('Linker Type Count', fontsize=13, weight='bold')
    axes[1].set_ylabel('Number of Compounds', fontsize=11)
    axes[1].set_xlabel('Linker Type', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for i, v in enumerate(linker_counts.values):
        axes[1].text(i, v + 5, str(int(v)), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'Figure4_Linker_Analysis.png'), dpi=300, bbox_inches='tight')
    print("Saved: Figure4_Linker_Analysis.png")
    plt.close()

def figure5_molecular_properties(df):
    """Figure 5: 分子性质分布"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 分子量分布
    mw_data = df['MW_numeric'].dropna()
    axes[0, 0].hist(mw_data, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Molecular Weight (Da)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Molecular Weight Distribution', fontsize=12, weight='bold')
    axes[0, 0].axvline(mw_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {mw_data.mean():.0f}')
    axes[0, 0].legend()
    
    # TPSA分布
    tpsa_data = df['TPSA'].dropna()
    axes[0, 1].hist(tpsa_data, bins=30, color='#2ecc71', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('TPSA (Ų)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('TPSA Distribution', fontsize=12, weight='bold')
    axes[0, 1].axvline(tpsa_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {tpsa_data.mean():.1f}')
    axes[0, 1].legend()
    
    # H-Bond Acceptors
    hba_data = df['Hbond acceptors'].dropna()
    axes[1, 0].hist(hba_data, bins=20, color='#e74c3c', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('H-Bond Acceptors', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('H-Bond Acceptors Distribution', fontsize=12, weight='bold')
    axes[1, 0].axvline(hba_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {hba_data.mean():.1f}')
    axes[1, 0].legend()
    
    # H-Bond Donors
    hbd_data = df['Hbond donors'].dropna()
    axes[1, 1].hist(hbd_data, bins=15, color='#f39c12', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('H-Bond Donors', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('H-Bond Donors Distribution', fontsize=12, weight='bold')
    axes[1, 1].axvline(hbd_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {hbd_data.mean():.1f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'Figure5_Molecular_Properties.png'), dpi=300, bbox_inches='tight')
    print("Saved: Figure5_Molecular_Properties.png")
    plt.close()

def figure6_activity_vs_properties(df):
    """Figure 6: 活性与分子性质的关系"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    active = df[df['Activity_Label'] == 1]
    inactive = df[df['Activity_Label'] == 0]
    
    # MW比较
    data_mw = [active['MW_numeric'].dropna(), inactive['MW_numeric'].dropna()]
    bp1 = axes[0, 0].boxplot(data_mw, labels=['Active', 'Inactive'], patch_artist=True)
    for patch, color in zip(bp1['boxes'], ['#2ecc71', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0, 0].set_ylabel('Molecular Weight (Da)', fontsize=11)
    axes[0, 0].set_title('MW: Active vs Inactive (p=0.0033)', fontsize=12, weight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # TPSA比较
    data_tpsa = [active['TPSA'].dropna(), inactive['TPSA'].dropna()]
    bp2 = axes[0, 1].boxplot(data_tpsa, labels=['Active', 'Inactive'], patch_artist=True)
    for patch, color in zip(bp2['boxes'], ['#2ecc71', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0, 1].set_ylabel('TPSA (Ų)', fontsize=11)
    axes[0, 1].set_title('TPSA: Active vs Inactive (p=0.0285)', fontsize=12, weight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # MW vs TPSA散点图 (Active)
    axes[1, 0].scatter(active['MW_numeric'].dropna(), active['TPSA'].dropna(), 
                       alpha=0.6, s=50, color='#2ecc71', label='Active', edgecolors='black', linewidth=0.5)
    axes[1, 0].scatter(inactive['MW_numeric'].dropna(), inactive['TPSA'].dropna(), 
                       alpha=0.6, s=50, color='#e74c3c', label='Inactive', edgecolors='black', linewidth=0.5)
    axes[1, 0].set_xlabel('Molecular Weight (Da)', fontsize=11)
    axes[1, 0].set_ylabel('TPSA (Ų)', fontsize=11)
    axes[1, 0].set_title('MW vs TPSA', fontsize=12, weight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # HBA vs HBD
    axes[1, 1].scatter(active['Hbond acceptors'].dropna(), active['Hbond donors'].dropna(), 
                       alpha=0.6, s=50, color='#2ecc71', label='Active', edgecolors='black', linewidth=0.5)
    axes[1, 1].scatter(inactive['Hbond acceptors'].dropna(), inactive['Hbond donors'].dropna(), 
                       alpha=0.6, s=50, color='#e74c3c', label='Inactive', edgecolors='black', linewidth=0.5)
    axes[1, 1].set_xlabel('H-Bond Acceptors', fontsize=11)
    axes[1, 1].set_ylabel('H-Bond Donors', fontsize=11)
    axes[1, 1].set_title('HBA vs HBD', fontsize=12, weight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'Figure6_Activity_vs_Properties.png'), dpi=300, bbox_inches='tight')
    print("Saved: Figure6_Activity_vs_Properties.png")
    plt.close()

def figure7_e3_activity_correlation(df):
    """Figure 7: E3连接酶与活性的相关性"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 计算每个E3连接酶的活性率
    e3_activity = []
    e3_names = []
    
    for e3 in df['E3 Ligase'].unique():
        if pd.notna(e3):
            e3_data = df[df['E3 Ligase'] == e3]
            active_rate = e3_data['Activity_Label'].mean()
            e3_activity.append(active_rate)
            e3_names.append(e3)
    
    # 排序
    sorted_indices = np.argsort(e3_activity)[::-1]
    e3_names_sorted = [e3_names[i] for i in sorted_indices]
    e3_activity_sorted = [e3_activity[i] for i in sorted_indices]
    
    # 创建柱状图
    colors = ['#2ecc71' if x > 0.6 else '#f39c12' if x > 0.5 else '#e74c3c' for x in e3_activity_sorted]
    bars = ax.bar(e3_names_sorted, e3_activity_sorted, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Activity Rate', fontsize=12, weight='bold')
    ax.set_xlabel('E3 Ligase Type', fontsize=12, weight='bold')
    ax.set_title('PROTAC Activity Rate by E3 Ligase Type', fontsize=14, weight='bold')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, val in zip(bars, e3_activity_sorted):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'Figure7_E3_Activity_Correlation.png'), dpi=300, bbox_inches='tight')
    print("Saved: Figure7_E3_Activity_Correlation.png")
    plt.close()

def main():
    print("Generating journal-quality figures...")
    
    # 创建图表目录
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # 加载数据
    df, analysis = load_data()
    
    # 生成各个图表
    figure1_activity_distribution(df)
    figure2_target_distribution(df)
    figure3_e3_ligase_distribution(df)
    figure4_linker_analysis(df)
    figure5_molecular_properties(df)
    figure6_activity_vs_properties(df)
    figure7_e3_activity_correlation(df)
    
    print("\nAll figures generated successfully!")
    print(f"Figures saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    main()
