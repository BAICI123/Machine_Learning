import matplotlib.pyplot as plt #仅用于画图
import GD_softmax as sf
import GD_logistic as ls
import standard as st

def softmax(X, Y, step, iterations, target=1e-3):
    standard_X = sf.st.standardize(X)
    one_hot_Y = sf.to_one_hot(Y)
    loss_history = []
    n = len(X[0])
    m = len(one_hot_Y[0])  # 取得类别的数量
    weights = [[0.0] * n for _ in range(m)]
    epochs = 0
    index = 0
    for i in range(iterations):

        # index+=1
        # if(index == len(X)):
        #     index = 0
        # x_i = [standard_X[index]]  # 当前样本
        # y_i = [one_hot_Y[index]]   # 当前样本的标签
        # new_weights = sf.update_weights(x_i, y_i, weights, step)
        # weights = new_weights  # 直接更新权重

        old_weights = weights
        import random
        indices = list(range(len(standard_X)))  # 所有样本的索引
        random.shuffle(indices)  # 打乱索引以实现随机梯度下降
        for index in indices:  # 遍历每个样本
            x_i = [standard_X[index]]  # 当前样本
            y_i = [one_hot_Y[index]]   # 当前样本的标签
            new_weights = sf.update_weights(x_i, y_i, weights, step)
            weights = new_weights  # 直接更新权重

        # epochs += 5
        # if(epochs > len(X)): epochs = epochs - len(X)
        # x_epoch = []
        # y_epoch = []
        # for epoch in range(epochs):
        #     x_epoch.append(standard_X[epoch])
        #     y_epoch.append(one_hot_Y[epoch])
        #     new_weights = sf.update_weights(x_epoch, y_epoch, weights, step)
        #     weights = new_weights


        flag = 1  
        for j in range(m):
            for k in range(n):
                if abs(new_weights[j][k] - old_weights[j][k]) > target:
                    flag = 0 
                    break 
            if(flag == 0):
                break
        if flag:  
            print(f"满足要求，iterations： {i + 1} .")
            break


        loss = sf.calculate_loss(standard_X, one_hot_Y, weights)
        print(f"iteration:{i + 1}, loss:{loss:.5f}, weights:{weights}")

        loss_history.append(loss)

        # if i%10 == 0:
        #     sf.plot_decision_boundary(standard_X, Y, weights, i + 1)


    return weights, loss_history

if __name__ == "__main__":
    X,Y = ls.take_file()
    weights,loss_history = softmax(X,Y,0.01,500)

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

    predict = sf.prediction(st.standardize(test_x),weights)
    print(f"predict:{predict}")