# 配置文件
import time

# 常用
num_epoch = 1  # 训练轮数
learning_rate = 0.0001  # 初始学习率
gamma = 1  # 学习率衰减指数
k = 3  # 颜色数
print_param = True  # 输出神经网络参数

GPU = True  # 要不要放到GPU上
use_max = True  # 用max代替add
change_cri = True  # True：用是否存在路径代替路径数量

use_ratio = 0.001  # 使用数据的百分比
train_ratio = 0.8  # 训练样本的百分比
batch_size = 1  # 批量训练大小，先全分为一组


file_name = "data"+str(k)+".txt"  # 生成/读取样例文件名
num_node_features = k * (1 << k)  # 节点特征数
randomSeed = time.time()  # time.time()  # 随机数种子，分别用于生成样例和划分训练集
draw_data = False  # 是否画出data
data_id = 0  # 画出data的id
print_interval = 1  # 每多少个epoch输出一次当前状态
