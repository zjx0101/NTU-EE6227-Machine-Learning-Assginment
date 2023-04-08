import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

class TreeNode:
    def __init__(self, min_attr, condition, gini, num, value, deter_class):
        self.attr = min_attr
        self.con = condition
        self.gini = gini
        self.num = num
        self.value = value
        self.deter_cl = deter_class
        self.left = None
        self.right = None

class ClassTree:
    def __init__(self, data):
        self.data = data
        min_attr, condition, gini_min, n, value, deter_class, data_l, data_r = self.cal_node_value(data)
        self.root = TreeNode(min_attr, condition, gini_min, n, value, deter_class)
        self.cur = self.root
        self.build_tree(data_l, data_r, self.root)

    def cal_gini(self, p1, p2, p3):
        return (1 - p1**2 - p2**2 - p3**2)/2

    def cal_attr(self, data, attr_num):
        sorted_data = np.array(sorted(data, key = lambda x:x[attr_num]))
        attr_mid = []
        gini_mid = []
        n = len(sorted_data)
        for i in range(n-1):
            attr_mid.append((sorted_data[i][attr_num]+sorted_data[i+1][attr_num])/2)
        for idx, mid in enumerate(attr_mid):
            p11, p12, p13, p21, p22, p23 = 0, 0, 0, 0, 0, 0
            for i in range(idx+1):
                if sorted_data[i][4]==1:
                    p11 += 1
                elif sorted_data[i][4]==2:
                    p12 += 1
                else:
                    p13 += 1
            for i in range(idx+1, n):
                if sorted_data[i][4]==1:
                    p21 += 1
                elif sorted_data[i][4]==2:
                    p22 += 1
                else:
                    p23 += 1
            p11, p12, p13 = p11/(idx+1), p12/(idx+1), p13/(idx+1)
            p21, p22, p23 = p21/(n-idx-1), p22/(n-idx-1), p23/(n-idx-1)
            gini_mid.append((idx+1)/n*self.cal_gini(p11,p12,p13) + (n-idx-1)/n*self.cal_gini(p21,p22,p23))
        
        gini_min = 0.5
        for idx, gini in enumerate(gini_mid):
            if gini<gini_min:
                idx_min = idx
                gini_min = gini
        return sorted_data[:idx_min+1], sorted_data[idx_min+1:], gini_min, attr_mid[idx_min], idx_min

    def cal_node_value(self, data):
        gini_min = 0.5
        n = data.shape[0]
        value = [0,0,0]
        for cl in data[:,4]:
            if cl==1:
                value[0] += 1
            elif cl==2:
                value[1] += 1
            else:
                value[2] += 1
        deter_class = value.index(max(value))+1

        for attr_i in range(4):
            data_l_local, data_r_local, gini_min_local, gini_mid_local, idx_min_local = self.cal_attr(data, attr_i)
            if gini_min_local<gini_min:
                idx_min = idx_min_local
                min_attr = attr_i+1
                gini_min = gini_min_local
                data_l = data_l_local
                data_r = data_r_local
                condition = gini_mid_local
        return min_attr, condition, gini_min, n, value, deter_class, data_l, data_r

    
    def build_tree(self, data_l, data_r, tree):
        min_attr, condition, gini_min, n, value, deter_class, data_ll, data_lr = self.cal_node_value(data_l)
        if n-value[deter_class-1]>2:
            tree.left = TreeNode(min_attr, condition, gini_min, n, value, deter_class)
            self.build_tree(data_ll, data_lr, tree.left)
        else:
            tree.left = TreeNode(None, None, 0, n, value, deter_class)

        min_attr, condition, gini_min, n, value, deter_class, data_rl, data_rr = self.cal_node_value(data_r)
        if n-value[deter_class-1]>2:
            tree.right = TreeNode(min_attr, condition, gini_min, n, value, deter_class)
            self.build_tree(data_rl, data_rr, tree.right)
        else:
            tree.right = TreeNode(None, None, 0, n, value, deter_class)

def print_tree(tree):
    if tree:
        print(tree.deter_cl, tree.con, tree.attr)
        print('left')
        print_tree(tree.left)
        print('back')
        print('right')
        print_tree(tree.right)
        print('back')
    else:
        return 0
    
def test(data, root):
    if root.left:
        attr = root.attr - 1
        con = root.con
        sorted_data = np.array(sorted(data, key = lambda x:x[attr]))
        for idx in range(len(sorted_data)):
            if sorted_data[idx, attr]>con:
                break
        left_res = test(sorted_data[:idx], root.left)
        right_res = test(sorted_data[idx:], root.right)
        ans = np.concatenate((left_res, right_res), axis=0)
        return ans
    else:
        if len(data)==0:
            return None
        label = np.ones((data.shape[0],1)) * root.deter_cl
        data = np.concatenate((data,label),axis=1)
        return data

def plot_attr(data, attr1, attr2):
    plt.scatter(data[data[:, 4]==1][:, attr1], data[data[:, 4]==1][:, attr2], c='r')
    plt.scatter(data[data[:, 4]==2][:, attr1], data[data[:, 4]==2][:, attr2], c='g')
    plt.scatter(data[data[:, 4]==3][:, attr1], data[data[:, 4]==3][:, attr2], c='b')
    plt.xlabel(f'feature {attr1+1}')
    plt.ylabel(f'feature {attr2+1}')
    plt.show()

train_data = sio.loadmat('Data_Train.mat')['Data_Train']
train_label = sio.loadmat('Label_Train.mat')['Label_Train']
test_data = sio.loadmat('Data_Test.mat')['Data_test']

data = np.concatenate((train_data,train_label),axis=1)
cal = ClassTree(data)
# print_tree(cal.root)

ans = test(test_data, cal.root)

print_tree(cal.root)

plot_attr(data, 0, 1)
plot_attr(data, 2, 3)