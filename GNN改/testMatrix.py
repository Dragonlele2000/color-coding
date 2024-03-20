import numpy as np

from GNN.VisualizeMatrix import visualize_matrix

k = 4
color_neighbor = 2
color_self = 1
file_name1 = 'Analyze\\' + str(k) + '-conv1.lin_rel.weight.txt'
file_name2 = 'Analyze\\' + str(k) + '-conv1.lin_root.weight.txt'

out_file1 = 'Analyze\\' + str(k) + '-result1.txt'
out_file2 = 'Analyze\\' + str(k) + '-result2.txt'


def main():
    m_rel = np.loadtxt(file_name1)
    m_root = np.loadtxt(file_name2)
    neighbor = np.zeros(((1 << k) * k, 1))
    neighbor[(1 << k) * color_neighbor + (1 << color_neighbor)][0] = 1
    self = np.zeros(((1 << k) * k, 1))
    self[(1 << k) * color_self + (1 << color_self)][0] = 1
    #print(m_rel * neighbor + m_root * self)
    print(self)
    np.savetxt(out_file1, self, fmt="%.1f")
    visualize_matrix(self, out_file1)
    new_self = m_rel @ neighbor + m_root @ self
    new_self = np.where(new_self < 0, 0, new_self)
    print(new_self)
    np.savetxt(out_file2, new_self, fmt="%.1f")
    visualize_matrix(new_self, out_file2)



if __name__ == '__main__':
    main()
