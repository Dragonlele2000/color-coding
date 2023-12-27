Gnn.py 运行程序
config.py 修改配置
  主要参数：
	file_name = 'data3.txt'：选择k=3的数据集
	file_name = 'data4.txt'：选择k=4的数据集
	gamma：每训练20(配置参数print_interval)轮后学习率指数衰减
  inputFile = False：不会生成新的数据集，而是从指定file_name中读取数据集
  inputFile = True：会生成新数据集，并写入file_name（覆盖写）
  draw_data = True：可以画出第data_id个样例
