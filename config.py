# 配置文件
import time

# 常用
num_epoch = 100  # 训练轮数
learning_rate = 0.001  # 初始学习率
gamma = 1  # 学习率衰减指数
file_name = "data4.txt"  # 生成/读取样例文件名
k = 4  # 颜色数
print_param = True  # 输出神经网络参数

GPU = not True  # 要不要放到GPU上

use_max = True
change_cri = True   # True：用是否存在路径代替路径数量
num_instance = 10000  # 样例数
train_n = int(0.8 * num_instance)  # 训练样本数，其余作为验证
batch_size = 64  # 批量训练大小，先全分为一组
v_l = 10  # 顶点数下界
v_u = 30  # 顶点数上界
p = 0.1  # 生成每条边的概率
inputFile = not True  # 是否生成样例并输入文件（否则从文件中读取）
readFromFile = not inputFile  # 是否从文件读取样例（否则重新生成）

randomSeed = time.time()  # time.time()  # 随机数种子，分别用于生成样例和划分训练集
debug = not True  # 生成样例的debug1
debug2 = not True  # 生成样例的debug2
num_node_features = k * (1 << k)  # 节点特征数
draw_data = False  # 是否画出data
data_id = 0  # 画出data的id
print_interval = 1  # 每多少个epoch输出一次当前状态
