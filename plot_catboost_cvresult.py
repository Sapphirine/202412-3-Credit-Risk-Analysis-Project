import matplotlib.pyplot as plt
import seaborn as sns

# CV scores数据
cv_scores = [0.8534533000159422, 0.8530952653289036, 0.8594515920429446, 
             0.8585735885353387, 0.8546149518185808]
avg_score = 0.8558377395483421

# 设置绘图风格
# plt.style.use('seaborn')
plt.figure(figsize=(10, 6))

# 创建柱状图
folds = range(1, len(cv_scores) + 1)
bars = plt.bar(folds, cv_scores, alpha=0.8, color=sns.color_palette("husl", len(cv_scores)))

# 添加平均分数线
plt.axhline(y=avg_score, color='red', linestyle='--', label=f'Mean AUC: {avg_score:.4f}')

# 在柱子上添加具体数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

# 设置图表属性
plt.xlabel('Fold', fontsize=12)
plt.ylabel('AUC Score', fontsize=12)
plt.title('CatBoost Cross-Validation Results', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 设置y轴范围，使图表更清晰
plt.ylim(min(cv_scores) - 0.002, max(cv_scores) + 0.002)

# 设置x轴刻度
plt.xticks(folds)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('catboost_cv_results.png')
plt.close()