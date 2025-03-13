import torch
import torch.nn as nn
import torch.optim as optim
import function as f
import random
import standard as st

def take_file(i):
    if i == 0:
        X, Y = f.take_file()
    else:
        X, Y = f.take_test_file()

    # 标准化
    X_standard = st.standardize(X)
    X = [[1] + X_standard[i] for i in range(len(X_standard))]

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32).view(-1, 1)  # Y 需要是一个列向量

# 定义前馈神经网络
class SimpleFeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2):
        super(SimpleFeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, 1)  # 输出1个值，适合二分类
        # 使用 Sigmoid 作为激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return self.sigmoid(x)  # 返回 Sigmoid 激活的输出

def k_fold_cross_validation(X, Y, k, num_epochs=100, hidden_size_1=3, hidden_size_2=7, learning_rate=0.01):
    fold_results = []
    fold_models = []  # 用于存储每个折叠的模型
    fold_size = len(X) // k

    # 打乱数据
    indices = list(range(len(X)))
    random.shuffle(indices)

    # 交叉验证
    for fold in range(k):
        print(f'Fold {fold + 1}')
        
        # 数据集和验证集
        val_indices = indices[fold * fold_size:(fold + 1) * fold_size]
        train_indices = indices[:fold * fold_size] + indices[(fold + 1) * fold_size:]
        
        X_train = X[train_indices]
        Y_train = Y[train_indices]
        X_val = X[val_indices]
        Y_val = Y[val_indices]

        # 初始化
        input_size = len(X[0])  # 第一层特征数量
        model = SimpleFeedforwardNN(input_size, hidden_size_1, hidden_size_2)

        # 损失函数
        criterion = nn.BCELoss()  # 使用二元交叉熵损失
        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 训练模型
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            # 前传
            outputs = model(X_train)
            loss = criterion(outputs, Y_train)

            # 误差后传，进行优化
            loss.backward()
            optimizer.step()

        # 验证模型
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_predictions = (val_outputs >= 0.5).float()  # 将输出转换为0或1
            val_accuracy = (val_predictions.view(-1) == Y_val.view(-1)).sum().item() / len(Y_val)

        fold_results.append(val_accuracy)
        fold_models.append(model)  # 存储当前折叠的模型
        print(f'Validation Accuracy for Fold {fold + 1}: {val_accuracy:.4f}')

    # 选择验证准确率最高的模型
    best_fold_index = fold_results.index(max(fold_results))
    best_model = fold_models[best_fold_index]

    return best_model, fold_results

# 主程序
if __name__ == "__main__":
    # 加载数据
    X, Y = take_file(0)

    # 进行 5 倍交叉验证
    k = 5
    model, fold_results = k_fold_cross_validation(X, Y, k)

    # 输出平均准确率
    avg_accuracy = sum(fold_results) / len(fold_results)
    print(f'Average Validation Accuracy: {avg_accuracy:.4f}')

    # 对测试集进行评估
    X_test, Y_test = take_file(1)  # 获取测试集
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        test_outputs = model(X_test)
        test_predictions = (test_outputs >= 0.5).float()  # 将输出转换为0或1
        test_accuracy = (test_predictions.view(-1) == Y_test.view(-1)).sum().item() / len(Y_test)  # 计算测试集准确率

    print(f'Test Accuracy: {test_accuracy:.4f}')
    print('Predicted:')
    print(test_predictions.view(-1))  # 打印预测的标签
