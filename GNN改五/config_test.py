# 配置文件
import time

# 常用
num_epoch = int(1e8)  # 训练轮数,1000
epoch_per_test = 20  # 训练多少轮输出一次测试结果
epoch_per_print = 10  # 训练多少轮输出一次参数
use_linear = not True  # 线性操作是否不固定
fix_root = True  # 是否固定root作为初始化

k = 6  # 颜色数
enable_drop = not True  # 把低于阈值的threshold设置为lower_bound，然后重新训练
# 否则不重新初始化
threshold = -10  # .1  # root元素低于此值即置为lower_bound
lower_bound = -1e8  # root元素需要到达多少才能达到选择效果
step = 1000  # 0.01  # 1  # .001  # .01  # 小于threshold时每一次下降的步长
step_mul = 1  # .01  #1.001  # 小于threshold时每一次下降的倍数
upper_bound = 1.1  # root元素高于此值即置为upper_bound
step_div = .99  # 高于upper_bound时每一次下降的倍数
tuning_rel = True  # rel元素<tuning_threshold时向正方向修正
tuning_threshold = -1e-4 #.01  # tuning_threshold修正下界
tuning_threshold2 = 1+1e-4  # tuning_threshold修正上界
tuning_step = 0 #1e-2  # 下界修正步长
tuning_step2 = 0  #.01  # 越过上界的修正步长
mul_tuning_step = 0 #1 越过下界的修正步长
mul_tuning_step2 = 1#0 越过上界的修正步长
learning_rate_rel = 1e-4  # 初始学习率
rate_ratio = 0 if fix_root else 1  # root_lr / rel_lr
use_ratio = 1  # 使用数据的百分比,0.005
train_ratio = .8  # 训练样本的百分比
batch_size = 128  # 批量训练大小,1
shuffle = True  # 训练时是否对每轮进行打乱

GPU = True  # 要不要放到GPU上
use_max = not True  # 用max代替add
change_cri = use_max  # True：用是否存在路径代替路径数量

print_param = True  # 输出神经网络参数
correct_standard_1 = 0.01  # 在此范围内认为结果正确
correct_standard_2 = 0.1  # 在此范围内认为结果正确
correct_standard_3 = 0.5  # 在此范围内认为结果正确
file_name = "data" + str(k) + ".txt"  # 生成/读取样例文件名
num_node_features = k * (1 << k)  # 节点特征数
randomSeed = time.time()  # time.time()  # 随机数种子，分别用于生成样例和划分训练集
draw_data = False  # 是否画出data
data_id = 0  # 画出data的id
print_interval = 1  # 每多少个epoch输出一次当前状态
