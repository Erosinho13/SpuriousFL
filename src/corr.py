import numpy as np
from sklearn.metrics import mutual_info_score

def AI(N):
    joint_prob = N / np.sum(N)
    P_feature = np.sum(joint_prob, axis=0)
    H_feature = -np.sum(P_feature * np.log(P_feature + 1e-9))
    nai = 1 - H_feature / np.log(len(P_feature))
    return nai


def CI(N):
    joint_prob = N / np.sum(N)
    P_class = np.sum(joint_prob, axis=1)
    H_class = -np.sum(P_class * np.log(P_class + 1e-9))
    nci = 1 - H_class / np.log(len(P_class))
    return nci


def SC(N):
    joint_prob = N / np.sum(N)
    P_class = np.sum(joint_prob, axis=1)
    P_feature = np.sum(joint_prob, axis=0)
    H_class = -np.sum(P_class * np.log(P_class + 1e-9))
    H_feature = -np.sum(P_feature * np.log(P_feature + 1e-9))
    mi = mutual_info_score(None, None, contingency=N)
    nmi = 2 * mi / (H_class + H_feature)
    return nmi


def global_N(M):
    return np.sum(M, axis=0)


def GSC(M):
    return SC(global_N(M))


def LSC(M):
    K = M.shape[0]
    return sum([SC(M[k]) for k in range(K)])/K


def FSC(M):
    E = []
    for N in M:
        row_sums = N.sum(axis=1, keepdims=True)
        F = N / row_sums
        E.append(F)
    K = M.shape[0]
    Y = M.shape[1]
    coef = 1 / (K * (K - 1) * Y)
    FSC = 0
    for i in range(K):
        for j in range(i + 1, K):
            FSC += np.sum(np.abs(E[i] - E[j]))
    FSC *= coef
    return FSC
    

def GCI(M):
    return CI(global_N(M))


def LCI(M):
    K = M.shape[0]
    return sum([CI(M[k]) for k in range(K)])/K


def GAI(M):
    return AI(global_N(M))


def LAI(M):
    K = M.shape[0]
    return sum([AI(M[k]) for k in range(K)]) / K


def main():

    M = np.array([

        [[10, 1990],
         [10, 1990]],

        [[1990, 10],
         [10, 1990]],

        [[1990, 10],
         [1990, 10]],

        [[10, 1990],
         [1990, 10]],

    ])

    # print(round(SC(N), 4))
    # print(round(CI(N), 4))
    # print(round(AI(N), 4))

    print(global_N(M))
    print(f"GSC = {round(GSC(M), 2)}")
    print(f"LSC = {round(LSC(M), 2)}")
    print(f"FSC = {round(FSC(M), 2)}")
    print(f"GCI = {round(GCI(M), 2)}")
    print(f"LCI = {round(LCI(M), 2)}")
    print(f"GAI = {round(GAI(M), 2)}")
    print(f"LAI = {round(LAI(M), 2)}")


if __name__ == '__main__':
    main()