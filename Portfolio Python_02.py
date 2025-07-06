import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 原始数据
data = {
    'Age': [22,23,24,24,25,26,26,27,27,28,28,29,31,31,32,32,34,34,35,35,36,36,38,39,39,41,42,43,44,45,46,47,48,49,50,51,52,54,55,57,58,61,62,64,66,67,69,70,72,74],
    'BMI': [24.666,25.798,25.515,26.453,23.539,22.481,30.620,26.289,19.662,28.972,30.381,22.191,21.167,20.369,29.181,23.524,27.299,20.290,27.325,20.605,24.237,30.964,24.730,29.734,27.958,29.679,27.369,21.520,26.877,39.080,25.394,26.715,23.562,26.589,24.772,37.587,25.917,26.851,26.735,24.887,27.594,29.230,29.290,22.595,25.104,25.767,26.631,24.497,24.628,29.770],
    'Waist': [34.606,34.882,39.370,37.165,34.843,30.551,40.512,35.787,28.661,38.898,41.339,33.268,30.079,31.969,38.386,34.921,37.953,29.370,37.953,29.528,35.551,42.835,36.063,41.142,41.181,43.031,37.047,32.087,39.803,49.685,37.638,37.008,36.220,41.969,35.197,48.071,40.000,39.370,37.520,38.071,39.291,38.701,41.024,32.598,38.819,35.315,37.402,37.362,39.291,42.717],
    'Pct.BF': [25.3,11.7,28.7,20.9,12.4,9.4,19.6,7.8,10.1,15.2,31.2,5.7,15.6,9.4,22.9,11.9,29.0,7.9,22.1,0.7,16.9,25.3,9.6,16.9,26.6,23.6,15.9,13.9,27.1,34.5,20.5,22.2,20.4,20.4,26.1,47.5,18.3,22.6,10.9,16.1,20.4,24.6,28.0,12.4,18.8,26.6,22.2,27.0,27.0,31.9],
    'Neck': [34,42.1,34.4,39,37.8,35.4,41.8,39.4,34.1,41.3,38.5,37.3,33.9,35,42.1,38.7,38.9,36.2,40.5,34,38.7,41.5,37.5,42.8,40,41.9,40.7,35.7,37.8,43.2,37.2,40,37,40.8,37.8,41.2,42,39.9,41.1,39.4,39.1,38.4,40.5,37.9,37.4,36.5,38.7,38.7,38.5,40.8]
}
df = pd.DataFrame(data)

# 添加年龄组列
df['Age_Group'] = df['Age'].apply(lambda x: '<40' if x < 40 else '≥40')

# 改进的Welch t检验函数
def welch_ttest(x, y, alpha=0.01):
    n1, n2 = len(x), len(y)
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
    
    t = (np.mean(x) - np.mean(y)) / np.sqrt(var1/n1 + var2/n2)
    df = (var1/n1 + var2/n2)**2 / (var1**2/(n1**2*(n1-1)) + var2**2/(n2**2*(n2-1)))
    
    p = 2 * (1 - stats.t.cdf(abs(t), df))
    se = np.sqrt(var1/n1 + var2/n2)
    margin = stats.t.ppf(1-alpha/2, df) * se
    ci = (np.mean(x)-np.mean(y) - margin, np.mean(x)-np.mean(y) + margin)
    
    return t, p, ci, df

# 执行检验
results = []
for var in ['BMI', 'Waist', 'Pct.BF', 'Neck']:
    t_stat, p_value, ci, df_val = welch_ttest(
        df[df['Age'] < 40][var], 
        df[df['Age'] >= 40][var]
    )
    conclusion = 'H₁' if p_value < 0.01 else 'H₀'
    
    results.append({
        'Variable': var,
        'Young_Mean': round(df[df['Age'] < 40][var].mean(), 2),
        'Old_Mean': round(df[df['Age'] >= 40][var].mean(), 2),
        'Mean_Diff': round(df[df['Age'] < 40][var].mean() - df[df['Age'] >= 40][var].mean(), 2),
        '99%_CI': (round(ci[0],2), round(ci[1],2)),
        't-statistic': round(t_stat, 3),
        'df': round(df_val, 1),
        'p-value': f"{p_value:.4f}",
        'Conclusion': conclusion
    })

result_df = pd.DataFrame(results)

# 创建绘图用的melt数据
melt_df = df.melt(
    id_vars=['Age_Group'], 
    value_vars=['BMI', 'Waist', 'Pct.BF', 'Neck'],
    var_name='Measurement',
    value_name='Value'
)

# 绘制箱线图
plt.figure(figsize=(12, 6))
ax = sns.boxplot(
    data=melt_df,
    x='Measurement',
    y='Value',
    hue='Age_Group',
    palette={'<40': '#1f77b4', '≥40': '#ff7f0e'}
)

# 添加统计标记
for i, row in result_df.iterrows():
    if row['Conclusion'] == 'H₁':
        ax.text(i, df[row['Variable']].max()*1.05, 
               f"p={float(row['p-value']):.3f}", 
               ha='center', fontsize=12, color='red')

plt.title('Body Measurements by Age Group\n( p < 0.01)', pad=20)
plt.ylabel('Value')
plt.xlabel('')
plt.legend(title='Age Group')
plt.tight_layout()
plt.show()

# 输出结果表
print(result_df.to_markdown(index=False))