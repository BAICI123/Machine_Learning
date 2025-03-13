# 计算每列的均值
def calculate_mean(data):
    n = len(data)  # 行数
    num_features = len(data[0])  # 列数
    means = []
    
    for j in range(num_features):
        column_sum = sum(row[j] for row in data)
        means.append(column_sum / n)
    
    return means

# 计算每列的标准差
def calculate_std(data, means):
    n = len(data)
    num_features = len(data[0])
    stds = []
    
    for j in range(num_features):
        variance_sum = sum((row[j] - means[j]) ** 2 for row in data)
        stds.append((variance_sum / n) ** 0.5)  # 计算标准差
    
    return stds

# 标准化数据
def standardize(data):
    means = calculate_mean(data)
    stds = calculate_std(data, means)
    
    standardized_data = []
    for row in data:
        standardized_row = [(row[j] - means[j]) / stds[j] for j in range(len(row))]
        standardized_data.append(standardized_row)
    
    return standardized_data

# def antistandard(data,weights):
#     n = len(weights)
#     means = calculate_mean(data)
#     stds = calculate_std(data, means)
#     weights_antistandard = [weights[i] if i == 0 else weights[i] * stds[i] for i in range(n)]
#     print(weights_antistandard)
#     return weights_antistandard
#计算每列的均值

if __name__ == "__main__":
    # 执行标准化
    data = [
    [2104, 5],
    [1600, 3],
    [2400, 4],
    [1416, 2],
    [3000, 4]
    ]
    normalized_data = standardize(data)

    # 输出结果
    print("标准化后的数据：")
    for row in normalized_data:
        print(row)
