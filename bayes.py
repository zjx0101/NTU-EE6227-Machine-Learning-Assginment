import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def cal_discriminant(mean, var, x, p):  # calculate the discriminant function
    d = mean.shape[0]
    y = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(var))) * np.exp(
        -1 / 2 * np.matmul(np.matmul((x - mean).T, np.linalg.inv(var)), (x - mean)))
    g = p * y
    return g


def cal_param(x):  # calculate mean and variance
    n = x.shape[0]
    mean = 1 / n * np.sum(x, axis=0)
    var = 1 / n * np.matmul((x - mean).T, (x - mean))
    return mean, var, n


def plot_attr(data, attr1, attr2, means):
    plt.scatter(data[data[:, 4]==1][:, attr1], data[data[:, 4]==1][:, attr2], c='r')
    plt.scatter(data[data[:, 4]==2][:, attr1], data[data[:, 4]==2][:, attr2], c='g')
    plt.scatter(data[data[:, 4]==3][:, attr1], data[data[:, 4]==3][:, attr2], c='b')
    plt.scatter(means[:, attr1], means[:, attr2], c='k')
    plt.xlabel(f'feature {attr1+1}')
    plt.ylabel(f'feature {attr2+1}')
    plt.show()


train_path = 'Data_Train.mat'
label_path = 'Label_Train.mat'
test_path = 'Data_test.mat'
train_data = sio.loadmat(train_path)['Data_Train']
label_data = sio.loadmat(label_path)['Label_Train']
test_data = sio.loadmat(test_path)['Data_test']

total_number = len(train_data)
label_1, label_2, label_3 = [], [], []

for i in range(len(train_data)): 
    if label_data[i] == 1:
        label_1.append(train_data[i])
    elif label_data[i] == 2:
        label_2.append(train_data[i])
    else:
        label_3.append(train_data[i])
label_1 = np.array(label_1)
label_2 = np.array(label_2)
label_3 = np.array(label_3)

mean1, var1, n1 = cal_param(label_1)
mean2, var2, n2 = cal_param(label_2)
mean3, var3, n3 = cal_param(label_3)
p1, p2, p3 = n1 / total_number, n2 / total_number, n3 / total_number

print('mean 1:\n', mean1)
print('variance 1:\n', var1)
print('mean 2:\n', mean2)
print('variance 2:\n', var2)
print('mean 3:\n', mean3)
print('variance 3:\n', var3)

res = []
for line in test_data:  # classify test_data lebel
    class_res1 = cal_discriminant(mean1, var1, line, p1)
    class_res2 = cal_discriminant(mean2, var2, line, p2)
    class_res3 = cal_discriminant(mean3, var3, line, p3)
    class_res = [class_res1, class_res2, class_res3]
    final_class = class_res.index(max(class_res)) + 1
    res.append(np.concatenate((line, np.array([final_class])), axis=0))
    
res = np.array(res)
means = [mean1,mean2,mean3]
means = np.array(means)
plot_attr(res, 0, 1, means)
plot_attr(res, 2, 3, means)