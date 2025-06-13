import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import gmean


def AI(N):
    # Attribute Inbalance
    joint_prob = N / np.sum(N)
    P_feature = np.sum(joint_prob, axis=0)
    H_feature = -np.sum(P_feature * np.log(P_feature + 1e-9))
    nai = 1 - H_feature / np.log(len(P_feature))
    return nai


def CI(N):
    # Class Inbalance
    joint_prob = N / np.sum(N)
    P_class = np.sum(joint_prob, axis=1)
    H_class = -np.sum(P_class * np.log(P_class + 1e-9))
    nci = 1 - H_class / np.log(len(P_class))
    return nci


def SC(N):
    # Spurious Correlation
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
    # Global Spurious Correlation
    return SC(global_N(M))


def LSC(M):
    # Local Spurious Correlation
    K = M.shape[0]
    return sum([SC(M[k]) for k in range(K)])/K


def FSC(M):
    # Federated Spurious Correlation
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
    # Global Class Imbalance
    return CI(global_N(M))


def LCI(M):
    # Local Class Imbalance
    K = M.shape[0]
    return sum([CI(M[k]) for k in range(K)])/K


def GAI(M):
    # Global Attribute Imbalance
    return AI(global_N(M))


def LAI(M):
    # Local Attribute Imbalance
    K = M.shape[0]
    return sum([AI(M[k]) for k in range(K)]) / K


def DSI(M):
    # Dataset Size Imbalance
    K = M.shape[0]
    n_images = np.sum(M, axis=(1, 2))
    tot_images = np.sum(n_images)
    proportion = n_images / tot_images
    return 1 - K * float(gmean(proportion))


def report_dist(M):

    

    # print(round(SC(N), 4))
    # print(round(CI(N), 4))
    # print(round(AI(N), 4))

    print(global_N(M))
    print(f"NoC = {round(len(M), 2)}")
    print(f"GSC = {round(GSC(M), 2)}")
    print(f"LSC = {round(LSC(M), 2)}")
    print(f"FSC = {round(FSC(M), 2)}")
    print(f"GCI = {round(GCI(M), 2)}")
    print(f"LCI = {round(LCI(M), 2)}")
    print(f"GAI = {round(GAI(M), 2)}")
    print(f"LAI = {round(LAI(M), 2)}")
    print(f"DSI = {round(DSI(M), 2)}")
    print("Total data:", np.sum(M))
    count = {"CI":0,"AI":0,"SC":0}
    for c in M:
        t = ["CI","AI","SC"][np.argmax(np.array([CI(c),AI(c),SC(c)]))]
        count[t]+=1
        #print(f"{CI(c):.2f},{AI(c):.2f},{SC(c):.2f}, {t}")
    print("Total:", count)

if __name__ == '__main__':
    M = np.array([

            [[10, 10],
            [10, 10]],

            [[10, 10],
            [10, 10]],

            [[10, 10],
            [10, 10]],

            [[10, 10],
            [10, 10]],

    ])
    report_dist(M)