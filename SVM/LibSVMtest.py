import function as f
import standard as st
from libsvm.svm import *
from libsvm.svmutil import *

def take_file(i):
    if i == 0:
        X, Y = f.take_file()
    else:
        X, Y = f.take_test_file()
    # 标准化
    X_standard = st.standardize(X)
    X = [[1] + X_standard[i] for i in range(len(X_standard))]
    return X,Y

# 训练数据
X_train, y_train = take_file(0)
# SVM 参数
params = '-s 0 -t 2 -c 1'  # C-SVC, RBF核, 惩罚参数C=1
# 将数据转换为 LibSVM 格式
problem = svm_problem(y_train, X_train)
# 训练 SVM 模型
model = svm_train(problem, params)

X_test, y_test = take_file(1)

#预测
p_label, p_acc, p_val = svm_predict(y_test, X_test, model)

print("预测标签:", p_label)
print("准确率:", p_acc)
