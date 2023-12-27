# 配置文件
import time

# 常用
num_epoch = 600  # 训练轮数
learning_rate = 0.01  # 初始学习率
gamma = 0.995  # 学习率衰减指数
inputFile = False  # 是否生成样例并输入文件（否则从文件中读取）
readFromFile = not inputFile  # 是否从文件读取样例（否则重新生成）
file_name = "data3.txt"  # 生成/读取样例文件名

num_instance = 1500  # 样例数
train_n = 1200  # 训练样本数，其余作为验证
k = 4  # 颜色数
batch_size = 2000  # 批量训练大小，先全分为一组
v_l = 5  # 顶点数下界
v_u = 12  # 顶点数上界
p = 0.3  # 生成每条边的概率

randomSeed = time.time()  # time.time()  # 随机数种子，分别用于生成样例和划分训练集
debug = not True  # 生成样例的debug1
debug2 = not True  # 生成样例的debug2
num_node_features = k * (1 << k)  # 节点特征数
draw_data = False  # 是否画出data
data_id = 0  # 画出data的id
print_interval = 20  # 每多少个epoch输出一次当前状态
