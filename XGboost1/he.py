import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取所有模型的预测结果，并使用 .squeeze("columns") 代替 squeeze 参数
rf_preds = pd.read_csv("rf_predictions.csv", header=None).squeeze("columns")
svm_preds = pd.read_csv("svm_predictions.csv", header=None).squeeze("columns")
xgb_preds = pd.read_csv("xgb_predictions.csv", header=None).squeeze("columns")
elm_preds = pd.read_csv("elm_predictions.csv", header=None).squeeze("columns")
dndt_preds = pd.read_csv("dndt_predictions.csv", header=None).squeeze("columns")
tabnet_preds = pd.read_csv("tabnet_predictions.csv", header=None).squeeze("columns")

# 创建包含所有模型预测结果的DataFrame
predictions_df = pd.DataFrame({
    'RandomForest': rf_preds,
    'SVM': svm_preds,
    'XGBoost': xgb_preds,
    'ELM': elm_preds,
    'DNDT': dndt_preds,
    'TabNet': tabnet_preds
})

# 计算皮尔逊相关系数矩阵
correlation_matrix = predictions_df.corr()

# 输出结果
print(correlation_matrix)

# 可视化皮尔逊相关系数矩阵，设置颜色条的范围从0.85开始
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=0.85, vmax=1, cbar_kws={'ticks': [0.85, 0.9, 0.95, 1]})
plt.title('Pearson Correlation Matrix of Model Predictions')
plt.savefig(f'pi.svg', format='svg', bbox_inches='tight')
plt.show()
