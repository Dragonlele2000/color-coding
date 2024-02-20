GenerateDataset/config_generate：
	生成数据集
	主要参数：
		k：颜色数

Gnn_test/config_test：
	训练网络，输出参数至Analyze文件夹
	主要参数：
		k：颜色数
		use_ratio：使用'datak.txt'中数据的比例
		train_ratio：训练样本比例
		batch_size：批量训练大小

VisualizeMatrix/config_visualize：
	可视化矩阵，从Analyze文件夹中读取对应矩阵
