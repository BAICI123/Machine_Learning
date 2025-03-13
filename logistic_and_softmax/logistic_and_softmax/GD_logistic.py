import matplotlib.pyplot as plt #仅用于画图
import standard as st

def exp(x):
    # 使用泰勒展开来近似计算 e^x
    term = 1.0 
    sum_exp = 1.0 
    n = 1
    while n < 100:
        term *= x / n
        sum_exp += term
        n += 1
    return sum_exp


def sigmoid(z):
    return 1 / (1 + exp(-z))


def log(x):
    if x <= 0:
        raise ValueError("Log(x),x为负值")
    # 使用牛顿法求解 ln(x)
    result = 0.0
    while True:
        guess = exp(result)  # 计算 e^result
        if abs(guess - x) < 1e-10:  # 收敛条件
            break
        result -= (guess - x) / guess  # 更新 guess
    return result


def calculate_loss(X, y, weights):
    m = len(y)
    loss = 0.0
    for i in range(m):
        h = sigmoid(sum(weights[j] * X[i][j] for j in range(len(weights))))
        loss += -y[i] * log(h) - (1 - y[i]) * log(1 - h)
    return loss


def plot_decision_boundary(X, y, weights, iteration):
    plt.figure()
    # 根据类别绘制样本点
    X = st.standardize(X)
    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(X[i][0], X[i][1], color='red', label='Class 1' if i == 0 else "")
        else:
            plt.scatter(X[i][0], X[i][1], color='blue', label='Class 0' if i == 0 else "")

    # 绘制决策边界
    x_min, x_max = min(x[0] for x in X) - 1, max(x[0] for x in X) + 1
    y_min, y_max = min(x[1] for x in X) - 1, max(x[1] for x in X) + 1

    xx = [x_min + i * 0.01 for i in range(int((x_max - x_min) / 0.01) + 1)]
    # 计算对应的决策边界
    # 由于权重长度增加，需要使用weights[0]作为偏置项
    yy = [-(weights[0] + weights[1] * x) / weights[2] for x in xx]  # 根据权重计算边界
    
    plt.plot(xx, yy, color='green', label='Decision Boundary')  # 绘制决策边界
    plt.title(f'Decision Boundary after Iteration {iteration}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid()
    plt.show()

   
def gradient_drop(X, y, step, max_iterations, tol=1e-3):
    m = len(y)
    n = len(X[0])
    weights = [0.0] * (n + 1)  # 初始化权重列表，增加一个偏置项
    temp = [0.0] * (n + 1)  # 存储梯度的临时列表
    loss_history = []  # 存储每次迭代的损失
    # 增加偏置项
    X_with_bias = [[1] + X_standard[i] for i in range(m)]  # 在每个样本前面添加1

    for iteration in range(max_iterations):
        weights_old = weights[:]
        temp = [0.0] * (n + 1)  # 每次迭代都重置临时列表
        for i in range(m):
            # 计算预测值，使用新的权重
            h = sigmoid(sum(weights[j] * X_with_bias[i][j] for j in range(n + 1)))  # 计算预测值
            for j in range(n + 1):  # 更新每个权重，包括偏置项
                temp[j] += (h - y[i]) * X_with_bias[i][j]
        for j in range(n + 1):  # 更新权重
            weights[j] -= step * temp[j]  # 计算平均梯度

        if all(abs(weights[j] - weights_old[j]) < tol for j in range(n + 1)):
            print(f"达到收敛条件，提前收敛，迭代次数为{iteration + 1}")
            break
        
        #计算损失并存储
        loss = calculate_loss(X_with_bias, y, weights)/m  # 使用包含偏置的特征计算损失
        loss_history.append(loss)  # 添加损失到历史记录
        if (iteration + 1) % 1 == 0:
            #绘制图像
            plot_decision_boundary(X, y, weights, iteration + 1)
            print(f"Iteration {iteration + 1}/{max_iterations}, Loss: {loss},weights:{weights}")

    return weights, loss_history  # 返回权重和损失历史


def predict(X, weights):
    X_standard = st.standardize(X)
    X_with_bias = [[1] + X_standard[i] for i in range(len(X))]  # 在每个样本前面添加1
    y_predicted = []
    for i in range(len(X)):
        classified = sigmoid(sum(weights[j] * X_with_bias[i][j] for j in range(len(weights))))  # 计算预测值
        y_predicted.append(1 if classified > 0.5 else 0)  # 进行二分类
    return y_predicted

def take_file():
    # 数据
    filename_1 = 'softmax作业数据/Exam/train/x.txt'  # 文件名
    X = []  # 特征数据
    with open(filename_1, 'r') as file:  # 以只读模式打开文件
        for line in file:
            # 拆分每一行并将其转换为浮点数，然后将其添加到 X 中
            X.append([float(num) for num in line.split()])

    Y = []  # 标签数据
    filename_2 = 'softmax作业数据/Exam/train/y.txt'
    with open(filename_2, 'r') as file:
        for line in file:
            Y.append(int(line.strip()))  # 添加标签数据（需要确保y.txt有标签数据）

    return X,Y


if __name__ == "__main__":
    
    X,Y = take_file()
    X_standard = st.standardize(X)
    #调用函数训练
    weights, loss_history = gradient_drop(X, Y, 0.1, 10,1e-3)

    print("weights:", weights)

    # 绘制损失图像
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss')
    plt.title('Loss over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    filename_3 = 'softmax作业数据/Exam/test/x.txt'  # 文件名
    test_x = []  # 特征数据
    with open(filename_3, 'r') as file:  # 以只读模式打开文件
        for line in file:
            # 拆分每一行并将其转换为浮点数，然后将其添加到 X 中
            test_x.append([float(num) for num in line.split()])

    predictions = predict(test_x, weights)
    print("预测：", predictions)
