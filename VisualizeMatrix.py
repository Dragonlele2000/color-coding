import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from config_visualize import *


def visualize_matrix(m, filename):
    cmap = ListedColormap(['#000000', '#666666','#CCCCCC', 'w','w',
                           'w', 'w', '#FF9999', '#CC3333', '#FF0000'])
    plt.rcParams['figure.figsize'] = [10, 16]
    # plt.rcParams['axes.axisbelow'] = True
    plt.matshow(m, cmap=cmap, interpolation='nearest',
                vmin=-1, vmax=1)

    x = m.shape[0]
    y = m.shape[1]
    plt.xticks(range(x))
    plt.yticks(range(y))
    plt.tick_params(axis='x', bottom=False)
    plt.grid(c='g', ls=':', lw='1')

    '''
    for i in range(x):
        for j in range(y):
            plt.text(j - 0.15, i, f"{m[i, j]:.1f}", color='white')
    '''
    plt.title(filename.split('/')[-1])
    plt.colorbar(ticks=np.linspace(-1, 1, len(cmap.colors)+1))
    plt.show()


def main():
    m1 = np.loadtxt(file_name1)
    m2 = np.loadtxt(file_name2)
    visualize_matrix(m1, file_name1)
    visualize_matrix(m2, file_name2)


if __name__ == '__main__':
    main()
