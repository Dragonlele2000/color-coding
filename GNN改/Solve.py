# 求解k-path
def get(v_i, S, v, e, color, t):
    if t[v_i][S] != -1:
        return t[v_i][S], t
    if S == (1 << color[v_i]):
        t[v_i][S] = 1
        return 1, t
    if (S >> color[v_i]) % 2 != 1:
        t[v_i][S] = 0
        return 0, t
    tempS = S - (1 << color[v_i])
    for v_j in range(v):
        if e[v_i][v_j] == 1:
            a, t = get(v_j, tempS, v, e, color, t)
            if a == 1:
                t[v_i][S] = 1
                return 1, t
    t[v_i][S] = 0
    return 0, t


def get2(v_i, S, v, e, color, t):
    if t[v_i][S] != -1:
        return t[v_i][S], t
    if S == (1 << color[v_i]):
        t[v_i][S] = 1
        return 1, t
    if (S >> color[v_i]) % 2 != 1:
        t[v_i][S] = 0
        return 0, t
    tempS = S - (1 << color[v_i])

    t[v_i][S] = 0
    for v_j in range(v):
        if e[v_i][v_j] == 1:
            a, t = get2(v_j, tempS, v, e, color, t)
            if a >= 1:
                t[v_i][S] += a
    return t[v_i][S], t


def solve(v, color, e, k):
    t = [[-1 for _ in range(1 << k)] for _ in range(v)]
    y = []
    positive = 0
    for i in range(v):
        # a,t = get(i, (1 << k) - 1, v, e, color,t)
        a, t = get2(i, (1 << k) - 1, v, e, color, t)
        y.append(a)
        if a >= 1:
            positive = 1

    return y, positive


if __name__ == "__main__":
    v = 5
    color = [0, 1, 2, 1, 2]
    e = [[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]]
    k = 3
    y, positive = solve(v, color, e, k)
    print(y)
