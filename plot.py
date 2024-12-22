import matplotlib.pyplot as plt

# 迭代次数和 AUC 分数
iterations = list(range(0, 6000, 100)) + [5996]
auc_scores = [
    0.6900318, 0.8181938, 0.8291931, 0.8356594, 0.8391455, 0.8411503,
    0.8426579, 0.8438076, 0.8446005, 0.8453084, 0.8459925, 0.8465424,
    0.8469455, 0.8473243, 0.8477378, 0.8481356, 0.8483924, 0.8486717,
    0.8489473, 0.8492242, 0.8494673, 0.8496776, 0.8499079, 0.8501190,
    0.8503100, 0.8505237, 0.8506436, 0.8507743, 0.8509316, 0.8511056,
    0.8512484, 0.8513605, 0.8514758, 0.8515619, 0.8516821, 0.8517819,
    0.8519067, 0.8520113, 0.8521378, 0.8522470, 0.8523432, 0.8524293,
    0.8525286, 0.8526003, 0.8526883, 0.8527725, 0.8528582, 0.8529241,
    0.8530073, 0.8530795, 0.8531279, 0.8531797, 0.8532490, 0.8532799,
    0.8533623, 0.8534151, 0.8534876, 0.8535164, 0.8535641, 0.8536241,
    0.8536786
]

# 绘制 AUC 曲线
plt.figure(figsize=(10, 6))
plt.plot(iterations, auc_scores, marker='o', linestyle='-', color='b', label='Test AUC')

# 设置图表信息
plt.xlabel('Iteration')
plt.ylabel('Test AUC')
plt.title('Training AUC Curve')
plt.legend()
plt.grid(True)
plt.show()
