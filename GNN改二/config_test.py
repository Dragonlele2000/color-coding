# 配置文件
import time

# 常用
num_epoch = int(1e8)  # 训练轮数,1000
epoch_per_test = 20  # 训练多少轮输出一次测试结果
epoch_per_print = 10  # 训练多少轮输出一次参数
epoch_change_lr = 10  # 训练多少轮改变一次学习率
learning_rate = 1e-2  # 初始学习率
min_rate = 1e-4  # 最小学习率
use_elu = not True  # 是否用leaky_relu代替relu
alpha = 1e-12  # leaky_relu的参数
use_linear = not True  # 线性操作是否不固定
use_ExpLR = False  # 是否使用ExpLR
gamma = 1  # 学习率衰减指数

k = 4  # 颜色数
print_param = True  # 输出神经网络参数

GPU = not True  # 要不要放到GPU上
use_max = True  # 用max代替add
change_cri = use_max  # True：用是否存在路径代替路径数量

use_ratio = 0.1  # 使用数据的百分比,0.005
train_ratio = 0.8  # 训练样本的百分比
batch_size = 1  # 批量训练大小,1
shuffle = True  # 训练时是否对每轮进行打乱


use_lr_loss = False  # 是否用loss作为学习率
max_loss_rate = 0.01  # 低于这个值就用loss作为学习率
lr_loss_ratio = 0.1  # 使用loss的多少作为学习率

correct_standard_1 = 0.01  # 在此范围内认为结果正确
correct_standard_2 = 0.1  # 在此范围内认为结果正确
correct_standard_3 = 0.5  # 在此范围内认为结果正确
file_name = "data"+str(k)+".txt"  # 生成/读取样例文件名
num_node_features = k * (1 << k)  # 节点特征数
randomSeed = time.time()  # time.time()  # 随机数种子，分别用于生成样例和划分训练集
draw_data = False  # 是否画出data
data_id = 0  # 画出data的id
print_interval = 1  # 每多少个epoch输出一次当前状态
