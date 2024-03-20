import time

k = 4
v_l = k  # 顶点数下界
v_u = 2 * k  # 顶点数上界
p = 0.5  # 生成每条边的概率
num_instance = 50000  # 样例数


file_name = "data"+str(k)+".txt"  # 生成/读取样例文件名
randomSeed = time.time()
debug = not True  # 生成样例的debug1
debug2 = not True  # 生成样例的debug2
