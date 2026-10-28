# ==========================================
# 第 1 部分：导入库与数据读取
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置画图风格
sns.set_theme(style="whitegrid")

# 读取数据（使用相对路径，因为你在 notebooks 文件夹下的 .py 脚本里运行）
# 这里的 '../data/...' 意思是：退回到上一级目录，进入 data 文件夹
df = pd.read_csv('../data/cs-training.csv', index_col=0)

print("================ 数据概览 ================")
print(f"数据形状: {df.shape}")
print("\n--- 前 5 行数据 ---")
print(df.head())
print("\n--- 数据类型与缺失情况 ---")
print(df.info())


# ==========================================
# 第 2 部分：目标变量分析 (作业问题 2)
# ==========================================
print("\n================ 目标变量分析 ================")
# 统计目标变量 'SeriousDlqin2yrs' 的类别数量和比例
target_counts = df['SeriousDlqin2yrs'].value_counts()
target_pct = df['SeriousDlqin2yrs'].value_counts(normalize=True) * 100

print(f"未违约 (0): {target_counts[0]} 条, 占比 {target_pct[0]:.2f}%")
print(f"违约 (1): {target_counts[1]} 条, 占比 {target_pct[1]:.2f}%")

# 画目标变量柱状图
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='SeriousDlqin2yrs')
plt.title('Target Variable Distribution (0=No Default, 1=Default)')
plt.show() # 显示图片


# ==========================================
# 第 3 部分：缺失值分析 (作业问题 3)
# ==========================================
print("\n================ 缺失值分析 ================")
# 统计每列的缺失数量和比例，并按比例降序排列
missing_data = df.isnull().sum().to_frame(name='Missing_Count')
missing_data['Missing_Percentage'] = (missing_data['Missing_Count'] / len(df)) * 100
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(by='Missing_Percentage', ascending=False)

print("存在缺失值的字段（按缺失比例降序）：")
print(missing_data)


# ==========================================
# 第 4 部分：探索变量分布与对比 (作业问题 4)
# ==========================================
print("\n================ 探索变量分布 (正在生成图表，请查看弹出的窗口或底部面板) ================")

# 定义需要分析的变量列表
cols_to_plot = ['age', 'MonthlyIncome', 'DebtRatio',
                'NumberOfTime30-59DaysPastDueNotWorse',
                'NumberOfTimes90DaysLate',
                'RevolvingUtilizationOfUnsecuredLines']

# --- 图 1：数值变量的直方图 ---
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
axes = axes.flatten()
fig.suptitle('Distribution of Variables', fontsize=16)

for i, col in enumerate(cols_to_plot):
    # 针对有极端值的变量限制范围以看清主体分布
    if col == 'MonthlyIncome':
        sns.histplot(df[col].dropna(), bins=50, ax=axes[i]).set(xlim=(0, 30000))
    elif col == 'RevolvingUtilizationOfUnsecuredLines':
        sns.histplot(df[col], bins=50, ax=axes[i]).set(xlim=(0, 1.5))
    else:
        sns.histplot(df[col], bins=50, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')

plt.tight_layout()
plt.show()

# --- 图 2：违约/未违约的对比图（箱线图） ---
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
axes = axes.flatten()
fig.suptitle('Variables vs Default (Boxplots)', fontsize=16)

for i, col in enumerate(cols_to_plot):
    if col == 'MonthlyIncome':
        sns.boxplot(data=df, x='SeriousDlqin2yrs', y=col, ax=axes[i]).set(ylim=(0, 30000))
    elif col == 'RevolvingUtilizationOfUnsecuredLines':
        sns.boxplot(data=df, x='SeriousDlqin2yrs', y=col, ax=axes[i]).set(ylim=(0, 1.5))
    else:
        sns.boxplot(data=df, x='SeriousDlqin2yrs', y=col, ax=axes[i])
    axes[i].set_title(f'{col} vs Default')

plt.tight_layout()
plt.show()

print("\n======== 代码运行完毕！请根据终端输出和图表，完成作业里的 Markdown 总结 ========")