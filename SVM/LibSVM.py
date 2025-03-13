from sklearn.model_selection import  GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import function as f
import standard as st

# 加载数据集
def take_file(i):
    if i == 0:
        X, Y = f.take_file()
    else:
        X, Y = f.take_test_file()
    # 标准化
    X_standard = st.standardize(X)
    X = [[1] + X_standard[i] for i in range(len(X_standard))]
    return X,Y


X_train, y_train = take_file(0)
X_test, y_test = take_file(1)


# param_grid =  {
#         'C': [0.01, 0.1, 1, 5, 10],   #惩罚参数过大过拟合，过小欠拟合
#         'kernel': ['linear']    # 线性核
#     }

# param_grid =  {
#         'C': [0.1, 5, 10, 15, 25],
#         'degree': [2, 4, 8, 10],  # 多项式核度数，决策复杂程度，过大过拟合，过小欠拟合
#         'kernel': ['poly']  # 多项式核
#     }

# param_grid =  {
#         'C': [0.1, 1, 3, 5, 10],
#         'gamma': [0.1, 0.5, 1, 5, 10],  # RBF核的gamma参数，过大过拟合，过小欠拟合
#         'kernel': ['rbf']   # RBF核
#     }

param_grid =  {
        'C': [0.01, 0.05, 0.1, 0.5, 1],
        'kernel': ['sigmoid']   # sigmoid核
    }

# 创建SVC模型
svc = SVC()

# 创建网格搜索对象
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, 
                           scoring='accuracy', cv=5, verbose=0, n_jobs=-1)

# 网格搜索
grid_search.fit(X_train, y_train)

# 最佳参数和最佳得分
print("最佳参数:", grid_search.best_params_)
print("最佳得分:", grid_search.best_score_)

# 使用最佳参数训练最终模型
best_model = grid_search.best_estimator_

#预测
y_pred = best_model.predict(X_test)

# 输出分类报告和准确率
print(classification_report(y_test, y_pred))
print("测试集准确率:", accuracy_score(y_test, y_pred))
