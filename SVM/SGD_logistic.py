import matplotlib.pyplot as plt
import function as ds
import standard as st
import random


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
def stochastic_gradient_descent(X, Y, step, iterations, tol=1e-3):
    weights = [0.0] * (len(X[0]) + 1)
    m = len(Y)
    X_standard = st.standardize(X)
    loss_history = []
    X_with_bias = [[1] + X_standard[i] for i in range(len(X_standard))]

    for iteration in range(iterations):
        begin_weights = weights
        for i in range(m):
            index = i
            grad = gradient(X_with_bias, Y, weights, index)
            weights = update_weights(weights, grad, step)

        loss = ds.calculate_loss(X_with_bias, Y, weights) / m
        loss_history.append(loss)

         # 计算权重变化的L2范数
        weight_change = sum((b - w) ** 2 for b, w in zip(begin_weights, weights)) ** 0.5
        if weight_change < tol:  # 判断权重变化是否小于容忍度
            print("已收敛")
            break
        print("未收敛")

    return weights, loss_history

def k_fold_cross_validation(X, Y, k=5, step=0.1, iterations=10):
    # 打乱数据集
    combined = list(zip(X, Y))
    random.shuffle(combined)
    X_shuffled, Y_shuffled = zip(*combined)

    fold_size = len(Y) // k
    losses = []
    all_weights = []
    all_accuracy = []
    flag = 0
    for i in range(k):

        print(f"Fold:{i+1}")

        # 划分训练集和验证集
        val_X = X_shuffled[i * fold_size: (i + 1) * fold_size]
        val_Y = Y_shuffled[i * fold_size: (i + 1) * fold_size]
        
        train_X = X_shuffled[:i * fold_size] + X_shuffled[(i + 1) * fold_size:]
        train_Y = Y_shuffled[:i * fold_size] + Y_shuffled[(i + 1) * fold_size:]

        # 执行随机梯度下降法
        weights, loss_history = stochastic_gradient_descent(train_X, train_Y, step, iterations)
        
        # 计算验证集的损失
        val_X_standard = st.standardize(val_X)
        val_X_with_bias = [[1] + val_X_standard[j] for j in range(len(val_X_standard))]
        loss = ds.calculate_loss(val_X_with_bias, val_Y, weights) / len(val_Y)

        # 计算准确率
        prediction = predict(val_X_with_bias,weights)
        right = 0
        for i in range (len(prediction)):
            if prediction[i] == val_Y[i]:
                right+=1
        
        if right/len(prediction) > flag:
            flag = right/len(prediction)
            bestweight = weights


        ds.plot_decision_boundary(val_X, val_Y, weights, i + 1)
        losses.append(loss)
        all_weights.append(weights)
        all_accuracy.append(right/(len(prediction)))

        print(f"Validation Loss: {loss}")
        print(f"Validation weights: {weights}")
        print(f"Validation accuracy: {right/len(prediction)}")

    avg_loss = sum(losses) / k
    avg_accuracy = sum(all_accuracy) / k
    avg_weights = [sum(all_weights[j][i] for j in range(k)) / k for i in range(len(weights))]
    print(f"Average Loss : {avg_loss}")
    print(f"Average weights : {avg_weights}")
    print(f"Average accuracy : {avg_accuracy}")

    return avg_weights,bestweight


def predict(X, weights):
    y_predicted = []
    for i in range(len(X)):
        classified = ds.sigmoid(dot_multi(X[i], weights))
        y_predicted.append(1 if classified > 0.5 else 0)
    return y_predicted

# 测试模型
if __name__ == "__main__":
    X, Y = ds.take_file()
    step = 0.01
    iterations = 100
    avg_weiths,bestweights = k_fold_cross_validation(X, Y, k=5, step=step, iterations=iterations)

    pX, pY = ds.take_test_file()
    pX_standard = st.standardize(pX)
    pX_with_bias = [[1] + pX_standard[i] for i in range(len(pX_standard))]
    prediction = predict(pX_with_bias,avg_weiths)
    bprediction = predict(pX_with_bias,bestweights)

    right = 0
    for i in range (len(prediction)):
        if prediction[i] == pY[i]:
            right+=1
    print("avg_weights_prediction:")
    print(f"predition: {prediction}")
    print(f"accuracy: {right/len(prediction)}")

    bright = 0
    for i in range (len(bprediction)):
        if bprediction[i] == pY[i]:
            bright+=1
    print("best_weights_prediction:")
    print(f"predition: {bprediction}")
    print(f"accuracy: {bright/len(bprediction)}")