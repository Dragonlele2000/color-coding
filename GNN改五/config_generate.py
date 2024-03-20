import time

k = 6  # (需要和config_test一致)
upper_bound = 5e10  # 路径数量上限
v_l = 10000#2 * k  # 参考值 2 * k  # 顶点数下界
v_u = 10000#16  # 参考值 k * k  # 顶点数上界
p = 0.0003  # 生成每条边的概率
num_instance = 1000  # 样例数

show_failure = True  # 是否显示生成失败样例
lower_bound = 2 * k  # 图中最小边数
print_epoch = 100  # 生成多少个样例输出一次
file_name = "data"+str(k)+".txt"  # (-2)生成/读取样例文件名
randomSeed = time.time()
debug = not True  # 生成样例的debug1
debug2 = not True  # 生成样例的debug2
