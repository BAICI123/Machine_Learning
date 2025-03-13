import matplotlib.pyplot as plt
import GD_logistic as ds
import standard as st


def dot_multi(x, weights):
    return sum(x[j] * weights[j] for j in range(len(weights)))

# 计算梯度
def gradient(X, Y, weights, index):
    h = ds.sigmoid(dot_multi(X[index], weights))
    error = h - Y[index]
    gradient = [error * X[index][j] for j in range(len(weights))]
    return gradient

# 更新权重
def update_weights(weights, gradient, step):
    for j in range(len(weights)):
        weights[j] -= step * gradient[j]
    return weights

# 随机梯度下降法
def stochastic_gradient_descent(X, Y, step, iterations,tol = 1e-3):
    weights = [0.0] * (len(X[0]) + 1)
    m = len(Y)
    print(m)
    X_standard = st.standardize(X)
    loss_history = []
    X_with_bias = [[1] + X_standard[i] for i in range(len(X_standard))]  # 修正此行
    for iteration in range(iterations):
        weights_old = weights
        for i in range(m):
            index = i
            grad = gradient(X_with_bias, Y, weights, index)
            weights = update_weights(weights, grad, step)
        
        if all(abs(weights_old[j] - weights[j]) < tol for j in range(len(X[0]) + 1)):
            print(f"达到收敛条件，提前收敛，迭代次数为{iteration + 1}")
            break

        loss = ds.calculate_loss(X_with_bias, Y, weights)/m
        loss_history.append(loss)
        if iteration % 1 == 0:
            ds.plot_decision_boundary(X,Y,weights,iteration+1)
            print(f"iterations {iteration}, Loss: {loss},weights:{weights}")

    return weights,loss_history

def predict(X, weights):
    y_predicted = []
    for i in range(len(X)):
        classified = ds.sigmoid(dot_multi(X[i], weights))
        y_predicted.append(1 if classified > 0.5 else 0)
    return y_predicted

# 测试模型
if __name__ == "__main__":
    X, Y = ds.take_file()
    step = 0.1
    iterations = 10
    weights,loss_history = stochastic_gradient_descent(X, Y, step, iterations)  
    print("Final weights:", weights)

    # 绘制损失图像
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss')
    plt.title('Loss over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    filename_3 = 'softmax作业数据/Exam/test/x.txt'
    test_x = []
    with open(filename_3, 'r') as file:
        for line in file:
            test_x.append([float(num) for num in line.split()])
    test_x = st.standardize(test_x)
    test_x_with_bias = [[1] + sample for sample in test_x]  # 添加偏置
    predictions = predict(test_x_with_bias, weights)  # 使用带偏置的测试数据
    print("预测：", predictions)
