import numpy as np
from VisualizeMatrix import visualize_matrix
from config_test import k, change_cri

def rel():
    arr = np.zeros((k * (1 << k), (k * (1 << k))))

    for color in range(k):
        istart = color * (1 << k)
        for i in range((1 << k)):
            for j in range((1 << k)):
                if (i >> color) % 2 == 1 and j == i - (1 << color) and j != 0:
                    for j2 in range(k):
                        arr[i + istart][j + (1 << k) * j2] = 1
    visualize_matrix(arr, 'Analyze\\IdealRel-' + str(k) + '.txt')
    return arr


def root():
    arr = np.zeros((k * (1 << k), (k * (1 << k)))) * (-1)
    # arr = np.identity(k * (1 << k))
    # print(arr)
    for color in range(k):
        j = color * (1 << k) + (1 << color)
        arr[j][j] = 1
        for i in range(k * (1 << k)):
            if not (color * (1 << k) <= i < (color + 1) * (1 << k)):
                arr[i][j] = -100000

    visualize_matrix(arr, 'Analyze\\IdealRoot-' + str(k) + '.txt')
    return arr


if __name__ == '__main__':
    rel()
    root()
