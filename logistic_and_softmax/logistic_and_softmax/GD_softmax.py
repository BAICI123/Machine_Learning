import matplotlib.pyplot as plt #仅用于画图
import GD_logistic as ls
import standard as st

def calculate_h(X, weights):
    m = len(X)
    num_classes = len(weights)
    h = []

    for i in range(m):
        z = [sum(weights[j][k] * X[i][k] for k in range(len(X[0]))) for j in range(num_classes)]
        exp_z = [ls.exp(zi) for zi in z]
        sum_exp_z = sum(exp_z)
        h.append([ei / sum_exp_z for ei in exp_z])  
    
    return h

def calculate_loss(X, Y, weights):
    m = len(Y)
    h = calculate_h(X, weights)
    loss = 0

    for i in range(m):
        for j in range(len(Y[0])):  # 对每个类别
            #样本i对类别j的真实标签one-hot   样本i预测为样本j的概率
            loss += -Y[i][j] * ls.log(h[i][j])  

    return loss / m  # 返回平均损失

def gradient(X, Y, weights):
    """计算所有权重的梯度"""
    n = len(X[0])  # 特征数量
    m = len(Y)     # 样本数量
    num_classes = len(weights)
    gradient_list = [[0] * n for _ in range(num_classes)]  # 初始化梯度列表

    h = calculate_h(X, weights)

    for i in range(m):
        for j in range(num_classes):
            for k in range(n):
                gradient_list[j][k] += (h[i][j] - Y[i][j]) * X[i][k]  # 更新梯度

    return gradient_list

def update_weights(X, Y, old_weights, step):
    gradients = gradient(X, Y, old_weights)
    new_weights = []

    for i in range(len(old_weights)):
        new_weights.append([old_weights[i][j] - step * gradients[i][j] for j in range(len(old_weights[i]))])  # 更新每个权重

    return new_weights

def to_one_hot(data):
    one_hot = []
    for value in data:
        if value == 1:
            one_hot.append([1, 0])  # 对应类别 1
        else:
            one_hot.append([0, 1])  # 对应类别 0
    return one_hot

def softmax(X, Y, step, iterations, target=1e-3):
    standard_X = st.standardize(X)
    one_hot_Y = to_one_hot(Y)
    loss_history = []
    n = len(X[0])
    m = len(one_hot_Y[0])  #取得类别的数量
    weights = [[0.0] * n for _ in range(m)]
    
    for i in range(iterations):
        new_weights = update_weights(standard_X, one_hot_Y, weights, step)
        loss = calculate_loss(standard_X, one_hot_Y, new_weights)
        print(f"iteration:{i + 1},loss:{loss:.5f},weights:{new_weights}")

        loss_history.append(loss)
        if i % 10 == 0:
            plot_decision_boundary(standard_X,Y,weights,i+1) 

        flag = 1  
        for j in range(m):
            for k in range(n):
                if abs(new_weights[j][k] - weights[j][k]) > target:
                    flag = 0 
                    break 
            if(flag == 0):
                break

        if flag:  
            print(f"满足要求，iterations： {i + 1} .")
            break

        weights = new_weights

    return weights, loss_history


def prediction(X,weights):
    m = len(X)
    n = len(weights)
    h = calculate_h(X,weights)
    predict = []
    predict_class = []
    for i in range(m):
        predict_class.append(h[i].index(max(h[i]))) # 两列，第一列的概率表示类别1，第二列概率表示类别0
    
    for value in predict_class:
        if(value == 0):
            predict.append(1)
        else: predict.append(0)

    return predict

def plot_decision_boundary(X, y, weights, iteration):
    plt.figure()
    # 根据类别绘制样本点
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
    yy = []
    for x in xx:
        # 当模型的概率相等时（0.5），决策边界出现
        z = [sum(weights[j][k] * x for k in range(len(weights[0]))) for j in range(len(weights))]
        # exp_z = [ls.exp(zi) for zi in z]
        # sum_exp_z = sum(exp_z)
        # probs = [ei / sum_exp_z for ei in exp_z]
        
        # 计算y值，使得第一个类的概率等于第二个类的概率
        # 这里假设线性决策边界，仅适用于二维情况
        if len(weights) == 2 and len(weights[0]) == 2:
            y_value = (weights[1][0] - weights[0][0]) * x + (weights[1][1] - weights[0][1]) / 2
            yy.append(y_value)
        else:
            yy.append(y_min)  # 如果不是二维情况，则不绘制

    plt.plot(xx, yy, color='green', label='Decision Boundary')  # 绘制决策边界
    plt.title(f'Decision Boundary after Iteration {iteration}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid()
    plt.show()



if __name__ == "__main__":
    X,Y = ls.take_file()
    weights,loss_history = softmax(X,Y,0.1,100)

    file_name = 'softmax作业数据/Exam/test/x.txt'
    test_x = []
    with open(file_name,'r') as file:
        for line in file:
            test_x.append([float(num) for num in line.split()])
    
    # 绘制损失图像
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss')
    plt.title('Loss over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    predict = prediction(st.standardize(test_x),weights)
    print(f"predict:{predict}")