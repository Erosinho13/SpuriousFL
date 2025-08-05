from typing import List
import numpy as np

from src.utils import list_to_matrix

def create_subsets_from_list(group_ids:dict, client_samples:List, rng=np.random.default_rng()) -> List[List]:
    """
    Inputs:
        group_ids: {'1':{'2':[3,4,5]}} format where '1' represents possible y keys 
            and '2' the possible spurious group ids
        client_samples: List of N matrixes for clients where the N matrix tells how many
            items of d[y][s] should go to the specific client
        rng: random generator
    Outputs:
        subsets: list of list of ids
    """
    subsets = [[] for _ in range(len(client_samples))]
    y_keys = group_ids.keys()
    s_keys = set()
    for inner_dict in group_ids.values():
        s_keys.update(inner_dict.keys())
    s_keys = list(s_keys)
    for y in y_keys:
        for s in s_keys:
            perm = rng.permutation(group_ids[y][s])
            cumulated_idx = 0
            for i,c in enumerate(client_samples):
                subset = perm[cumulated_idx:cumulated_idx+c[y][s]]
                subsets[i].extend(subset)
                cumulated_idx += c[y][s]
    return subsets

def split_mode_to_matrix(conf:dict)-> List:
    """Named split modes to N matrix list"""
    split_mode = conf["dataset_options"]["split_mode"]
    seed = conf["seed"]

    client_samples = None
    if split_mode=="balanced":
        num_clients = conf["dataset_options"]["num_clients"]
        num_groups = conf["dataset_options"]["num_groups"]
        num_targets = conf["dataset_options"]["num_targets"]
        client_samples = np.ones((num_clients,num_targets,num_groups), dtype=int) * 50
        client_samples = client_samples.tolist()
    if split_mode=="zeroCI_LSC":
        client_samples=[
            [ # Birds on land
                [23,5],
                [23,5]
            ],
            [ # Expected background
                [896,5],
                [5,896]
            ],
            [ # Unexpected background
                [5,23],
                [23,5]
            ],
            [ # Birds on water
                [5,151],
                [5,151]
            ]
        ]
    if split_mode=="nCI_LSC":
        client_samples=[
            [ # Birds on land
                [23,5],
                [23,5]
            ],
            [ # Expected background
                [3465,5],
                [5,896]
            ],
            [ # Unexpected background
                [5,23],
                [23,5]
            ],
            [ # Birds on water
                [5,151],
                [5,151]
            ]
        ]
    if split_mode=="nDI_nGSC":
        client_samples=[
            [ # Birds on land
                [23,5],
                [23,5]
            ],
            [ # Expected background
                [23,5],
                [5,23]
            ],
            [ # Unexpected background
                [5,23],
                [23,5]
            ],
            [ # Birds on water
                [5,23],
                [5,23]
            ]
        ]
    if split_mode=="LCI_LSC_noG":
        client_samples=[
            [ 
                [23,23],
                [5,5]
            ],
            [ 
                [23,5],
                [5,23]
            ],
            [ 
                [5,23],
                [23,5]
            ],
            [ 
                [5,5],
                [23,23]
            ]
        ]
    if split_mode=="more_expected_clients":
        client_samples=[
            [ # Birds on land
                [28,0],
                [28,0]
            ],
            [ # Expected background
                [28,0],
                [0,28]
            ],
            [ # Unexpected background
                [0,28],
                [28,0]
            ],
            [ # Birds on water
                [0,28],
                [0,28]
            ]
        ]
        for i in range(35):
            client_samples.append([[28,0],[0,28]])
    if split_mode=="CI_GSC_reverse": # global minority is local mayority
        client_samples=[
            [ # mayority land/land bird
                [400,5],
                [5,5]
            ],
            [ # mayority water/water bird
                [5,5],
                [5,400]
            ],
            [ # minority but class majority
                [40,40],
                [5,5]
            ],
            [ # minority but class majority
                [5,5],
                [40,40]
            ]
        ]
    if split_mode=="spawrious1":
        client_samples = []
        client_samples += [[[90, 90], [10, 10]]] * 3  # Mostly waterbirds, Type CI (0)
        client_samples += [[[10, 10], [90, 90]]] * 3  # Mostly landbirds, Type CI (0)
        client_samples += [[[90, 10], [90, 10]]] * 3  # Birds on land, Type AI (1)
        client_samples += [[[10, 90], [10, 90]]] * 3  # Birds on water, Type AI (1)
        client_samples += [[[90, 10], [10, 90]]] * 5  # Expected background, Type SC (2)
        client_samples += [[[10, 90], [90, 10]]] * 1  # Unexpected background, Type SC (2)

    if split_mode=="spawrious2":
        client_samples = []
        client_samples += [[[90, 90], [10, 10]]] * 2  # Mostly waterbirds, Type CI (0)
        client_samples += [[[10, 10], [90, 90]]] * 2  # Mostly landbirds, Type CI (0)
        client_samples += [[[90, 10], [90, 10]]] * 2  # Birds on land, Type AI (1)
        client_samples += [[[10, 90], [10, 90]]] * 2  # Birds on water, Type AI (1)
        client_samples += [[[90, 10], [10, 90]]] * 15  # Expected background, Type SC (2)
        client_samples += [[[10, 90], [90, 10]]] * 1  # Unexpected background, Type SC (2)
    if split_mode=="adult_gsc":
        client_samples = []
        client_samples += [[[260, 260], [10, 10]]] * 2
        client_samples += [[[10, 10], [260, 260]]] * 2
        client_samples += [[[260, 10], [260, 10]]] * 2
        client_samples += [[[10, 260], [10, 260]]] * 2
        client_samples += [[[260, 10], [10, 260]]] * 15
        client_samples += [[[10, 260], [260, 10]]] * 1
    if split_mode=="testdist":
        client_samples = []
        client_samples += [[[170,10],[19,1]]] * 2
        client_samples += [[[150,30],[17,3]]] * 2
        client_samples += [[[130,50],[15,5]]] * 2
        client_samples += [[[110,70],[13,7]]] * 2
        client_samples += [[[100,80],[11,9]]] * 2
        client_samples += [[[10,170],[19,1]]] * 2
        client_samples += [[[30,150],[17,3]]] * 2
        client_samples += [[[50,130],[15,5]]] * 2
        client_samples += [[[70,110],[13,7]]] * 2
        client_samples += [[[80,100],[11,9]]] * 2
    if split_mode=="testdist2":
        client_samples = []
        client_samples += [[[170,10],[60,20]]] * 2
        client_samples += [[[150,30],[56,24]]] * 2
        client_samples += [[[130,50],[52,28]]] * 2
        client_samples += [[[110,70],[48,32]]] * 2
        client_samples += [[[100,80],[44,36]]] * 2
        client_samples += [[[10,170],[60,20]]] * 2
        client_samples += [[[30,150],[56,24]]] * 2
        client_samples += [[[50,130],[52,28]]] * 2
        client_samples += [[[70,110],[48,32]]] * 2
        client_samples += [[[80,100],[44,36]]] * 2
    
    if split_mode=="class4":
        client_samples = []
        client_samples += [[[500,50],[500,50],[50,500],[50,500]]] * 2

    if split_mode=="spawrious_GCI":
        client_samples=[
            [[90, 90], [10, 10]],
            [[90, 90], [10, 10]],
            [[90, 90], [10, 10]],
            [[90, 90], [10, 10]],
            [[90, 90], [10, 10]],
            [[90, 90], [10, 10]],
            [[90, 90], [10, 10]],
            [[90, 90], [10, 10]],
            [[90, 90], [10, 10]],
            [[90, 90], [10, 10]],
            [[90, 90], [10, 10]],
            [[90, 90], [10, 10]],
            [[90, 90], [10, 10]],
            [[90, 90], [10, 10]],
            [[90, 90], [10, 10]],
            [[10, 10], [90, 90]],
            [[90, 10], [90, 10]],
            [[90, 10], [90, 10]],
            [[10, 90], [10, 90]],
            [[10, 90], [10, 90]],
            [[90, 10], [10, 90]],
            [[90, 10], [10, 90]],
            [[10, 90], [90, 10]],
            [[10, 90], [90, 10]]
        ]

    if split_mode=="spawrious_GAI":
        client_samples=[
            [[90, 90], [10, 10]],
            [[90, 90], [10, 10]],
            [[10, 10], [90, 90]],
            [[10, 10], [90, 90]],
            [[90, 10], [90, 10]],
            [[90, 10], [90, 10]],
            [[90, 10], [90, 10]],
            [[90, 10], [90, 10]],
            [[90, 10], [90, 10]],
            [[90, 10], [90, 10]],
            [[90, 10], [90, 10]],
            [[90, 10], [90, 10]],
            [[90, 10], [90, 10]],
            [[90, 10], [90, 10]],
            [[90, 10], [90, 10]],
            [[90, 10], [90, 10]],
            [[90, 10], [90, 10]],
            [[90, 10], [90, 10]],
            [[90, 10], [90, 10]],
            [[10, 90], [10, 90]],
            [[90, 10], [10, 90]],
            [[90, 10], [10, 90]],
            [[10, 90], [90, 10]],
            [[10, 90], [90, 10]]
         ]
    if split_mode=="waterbirds_dist":
        client_samples=[
            [[23,23],[10,110]],
            [[110,23],[10,23]],
            [[89,39],[1,29]],
            [[29,39],[1,89]],
            [[81,35],[9,31]],
            [[126,1],[1,31]],
            [[126,1],[1,31]],
            [[126,1],[1,31]],
            [[126,1],[1,31]],
            [[126,1],[1,31]],
            [[126,1],[1,31]],
            [[126,1],[1,31]],
            [[126,1],[1,31]],
            [[126,1],[1,31]],
            [[127,1],[1,31]],
            [[127,1],[1,31]],
            [[127,1],[1,31]],
            [[127,1],[1,31]],
            [[127,1],[1,31]],
            [[127,1],[1,31]],
            [[127,1],[1,31]],
            [[127,1],[1,31]],
            [[127,1],[1,31]],
            [[127,1],[1,31]],
            [[127,1],[1,31]],
            [[127,1],[1,31]],
            [[127,1],[1,31]],
            [[127,1],[1,31]],
            [[127,1],[1,31]],
            [[127,1],[1,31]],
         ]
    if split_mode=="waterbirds_dist_reversed":
        client_samples=[
            [[23, 23], [110, 10]],
            [[23, 110], [23, 10]],
            [[39, 89], [29, 1]],
            [[39, 29], [89, 1]],
            [[35, 81], [31, 9]],
            [[1, 126], [31, 1]],
            [[1, 126], [31, 1]],
            [[1, 126], [31, 1]],
            [[1, 126], [31, 1]],
            [[1, 126], [31, 1]],
            [[1, 126], [31, 1]],
            [[1, 126], [31, 1]],
            [[1, 126], [31, 1]],
            [[1, 126], [31, 1]],
            [[1, 127], [31, 1]],
            [[1, 127], [31, 1]],
            [[1, 127], [31, 1]],
            [[1, 127], [31, 1]],
            [[1, 127], [31, 1]],
            [[1, 127], [31, 1]],
            [[1, 127], [31, 1]],
            [[1, 127], [31, 1]],
            [[1, 127], [31, 1]],
            [[1, 127], [31, 1]],
            [[1, 127], [31, 1]],
            [[1, 127], [31, 1]],
            [[1, 127], [31, 1]],
            [[1, 127], [31, 1]],
            [[1, 127], [31, 1]],
            [[1, 127], [31, 1]]
        ]
    if split_mode=="spawrious4":
        client_samples = []
        client_samples += [[[20, 20], [20, 20], [5, 5], [5, 5]]] * 2  # CI
        client_samples += [[[5, 5], [5, 5], [20, 20], [20, 20]]] * 2  # CI
        client_samples += [[[20, 5], [20, 5], [20, 5], [20, 5]]] * 2  # AI
        client_samples += [[[5, 20], [5, 20], [5, 20], [5, 20]]] * 2  # AI
        client_samples += [[[5, 20], [5, 20], [20, 5], [20, 5]]] *1 # SC
        client_samples += [[[119, 5], [119, 5], [5, 119], [5, 119]]] * 7  # SC
        client_samples += [[[118, 5], [118, 5], [5, 118], [5, 118]]] * 9  # SC
    
    if split_mode=="spawrious_GAI_2":
        client_samples=[
            [[120, 5], [20, 10]],
            [[120, 40], [5, 10]],

            
            [[170, 5], [5, 5]],
            [[170, 5], [5, 5]],
            [[5, 5], [170, 5]],
            [[5, 5], [170, 5]],

            [[10, 30], [120, 10]],
            [[10, 30], [120, 10]],

            [[80, 80], [20, 2]],
            [[80, 80], [20, 2]],

            [[80, 15], [90, 2]],
            [[80, 15], [90, 2]],
            [[80, 15], [90, 2]],
            [[80, 15], [90, 2]],
            [[80, 15], [90, 2]],
            [[80, 15], [90, 2]],
            [[80, 15], [90, 2]],
            [[80, 15], [90, 2]],
            [[80, 15], [90, 2]],
            [[80, 15], [90, 2]],
            [[80, 15], [90, 2]],
            [[80, 15], [90, 2]],
            [[80, 15], [90, 2]],
            [[80, 15], [90, 2]],
            [[110,5],[85,8]]

        ]

    if split_mode=="spawrious_GSC_6":
        global_dist_target = np.array([
            [5500,5500,5500,500,500,500],
            [5500,5500,5500,500,500,500],
            [500,500,500,5500,5500,5500],
            [500,500,500,5500,5500,5500]
        ])
        num_clients = 100
        num_attributes = 6
        num_classes = 4
        alpha = 0.1
        min_samples = 2
        client_samples = generate_clients_with_global_params(global_dist_target,num_clients,num_attributes,num_classes,alpha,min_samples,seed)
    if split_mode=="spawrious_GSC_6_2":
        global_dist_target = np.array([
            [5500,5500,5500,5500,5500,300],
            [5500,3000,3000,3000,3000,300],
            [5500,3000,3000,3000,3000,300],
            [300,300,300,300,300,4000]
        ])
        num_clients = 100
        num_attributes = 6
        num_classes = 4
        alpha = 0.1
        min_samples = 2
        client_samples = generate_clients_with_global_params(global_dist_target,num_clients,num_attributes,num_classes,alpha,min_samples,seed)
    
    if split_mode=="spawrious_GCI_100":
        client_samples=[
            [[68, 68], [10, 10]],
            [[68, 68], [10, 10]],
            [[68, 68], [10, 10]],
            [[68, 68], [10, 10]],
            [[68, 68], [10, 10]],
            [[68, 68], [10, 10]],
            [[68, 68], [10, 10]],
            [[68, 68], [10, 10]],
            [[68, 68], [10, 10]],
            [[68, 68], [10, 10]],
            [[68, 68], [10, 10]],
            [[68, 68], [10, 10]],
            [[68, 68], [10, 10]],
            [[68, 68], [10, 10]],
            [[68, 68], [10, 10]],
            [[68, 68], [10, 10]],
            [[10, 10], [68, 68]],
            [[68, 10], [68, 10]],
            [[68, 10], [68, 10]],
            [[10, 68], [10, 68]],
            [[10, 68], [10, 68]],
            [[68, 10], [10, 68]],
            [[68, 10], [10, 68]],
            [[10, 68], [68, 10]],
            [[10, 68], [68, 10]]
        ]
        client_samples += client_samples + client_samples + client_samples
    if split_mode=="spawrious_GSC_100":
        client_samples = []
        client_samples += [[[68, 68], [10, 10]]] * 2  # Mostly waterbirds, Type CI (0)
        client_samples += [[[10, 10], [68, 68]]] * 2  # Mostly landbirds, Type CI (0)
        client_samples += [[[68, 10], [68, 10]]] * 2  # Birds on land, Type AI (1)
        client_samples += [[[10, 68], [10, 68]]] * 2  # Birds on water, Type AI (1)
        client_samples += [[[68, 10], [10, 68]]] * 16  # Expected background, Type SC (2)
        client_samples += [[[10, 68], [68, 10]]] * 1  # Unexpected background, Type SC (2)
        client_samples += client_samples + client_samples + client_samples
    if split_mode == "spawrious_GAI_100":
        client_samples=[
            [[60, 5], [20, 10]],
            [[60, 35], [5, 10]],

            
            [[80, 5], [5, 5]],
            [[80, 5], [5, 5]],
            [[5, 5], [80, 5]],
            [[5, 5], [80, 5]],

            [[10, 30], [60, 10]],
            [[10, 30], [60, 10]],

            [[60, 60], [20, 2]],
            [[60, 60], [20, 2]],

            [[60, 12], [70, 2]],
            [[60, 12], [70, 2]],
            [[60, 12], [70, 2]],
            [[60, 12], [70, 2]],
            [[60, 12], [70, 2]],
            [[60, 12], [70, 2]],
            [[60, 12], [70, 2]],
            [[60, 12], [70, 2]],
            [[60, 12], [70, 2]],
            [[60, 12], [70, 2]],
            [[60, 12], [70, 2]],
            [[60, 12], [70, 2]],
            [[60, 12], [70, 2]],
            [[60, 12], [70, 2]],
            [[60,5],[60,8]]

        ]
        client_samples += client_samples + client_samples + client_samples
        
    if split_mode=="celeba_gsc":
        client_samples = []
        client_samples += [[[9, 9], [1, 1]]] * 2  #  Type CI (0)
        client_samples += [[[1, 1], [9, 9]]] * 2  # Type CI (0)
        client_samples += [[[9, 1], [9, 1]]] * 2  # Type AI (1)
        client_samples += [[[1, 9], [1, 9]]] * 2  # Type AI (1)
        client_samples += [[[9, 1], [1, 9]]] * 1  # Type SC (2)
        client_samples += [[[1, 9], [9, 1]]] * 15  # Type SC (2)
    if split_mode=="celeba_gci":
        client_samples = []
        client_samples += [[[9, 9], [1, 1]]] * 15  #  Type CI (0)
        client_samples += [[[1, 1], [9, 9]]] * 1  # Type CI (0)
        client_samples += [[[9, 1], [9, 1]]] * 2  # Type AI (1)
        client_samples += [[[1, 9], [1, 9]]] * 2  # Type AI (1)
        client_samples += [[[9, 1], [1, 9]]] * 2  # Type SC (2)
        client_samples += [[[1, 9], [9, 1]]] * 2  # Type SC (2)
    if split_mode=="celeba_gai":
        client_samples = []
        client_samples += [[[9, 9], [1, 1]]] * 2  #  Type CI (0)
        client_samples += [[[1, 1], [9, 9]]] * 2  # Type CI (0)
        client_samples += [[[9, 1], [9, 1]]] * 15  # Type AI (1)
        client_samples += [[[1, 9], [1, 9]]] * 1  # Type AI (1)
        client_samples += [[[9, 1], [1, 9]]] * 2  # Type SC (2)
        client_samples += [[[1, 9], [9, 1]]] * 2  # Type SC (2)
    if split_mode=="celeba_4_no_gci":
        client_samples = [
            [[  21,  177,    7,    2],[  22,  827,   16,   23]],
            [[  25,  163,  258,   23],[   3,  292,   13,   38]],
            [[  21,   57,  228,  329],[   6,  232,   23,    4]],
            [[   2,  325,   32,  481],[   8,   94,    5,   12]],
            [[  39,    8,  347,   88],[  48,  216,    4,   10]],
            [[  13,  419,   44,  190],[   5,   79,   14,   16]],
            [[  54,   67,   81,   35],[  12,  293,   28,    8]],
            [[  12,  200,   73,   32],[  22,  116,   52,    5]],
            [[  48,   27,  160,    3],[  58,  375,   11,    2]],
            [[  17,   36,  115,    6],[  12,  342,   54,   18]],
            [[  47,  187,  100,  111],[  16,   16,   22,    5]],
            [[  13,  107,   72,  449],[  13,  103,    7,   29]],
            [[   2,    5,  218,  238],[   8,  327,   28,   35]],
            [[   9,   49,  123,   35],[  27,  175,   34,   46]],
            [[  13,    9,  141,  164],[   6, 1148,   10,   22]],
            [[  17,   14,   74,  140],[  36,  475,    8,   46]],
            [[  18,  117,   22,  199],[  36,  152,   32,    9]],
            [[  25,    8,   93,   60],[  10,  250,   24,   58]],
            [[   5,   60,  214,   72],[  22,  796,   15,   30]],
            [[  11,  340,   20,   15],[  22,  324,   27,   15]],
            [[  21,   80,  230,  129],[   6,  352,    8,   45]],
            [[  37,   23,  287,   63],[  42,  409,    6,   12]],
            [[  17,  153,    9,   11],[  47,  385,   35,    6]],
            [[  13,  369,   52,  125],[  13,  222,   24,    6]]
        ]
    if split_mode == "celeba_4_natural":
        client_samples = [
            [[ 20, 280,   6,   3],[ 26, 374,   9,  33]],
            [[ 27, 254, 188,  28],[  3, 133,   8,  52]],
            [[ 23,  89, 166, 402],[  7, 106,  13,   5]],
            [[  2, 510,  24, 588],[  9,  44,   4,  16]],
            [[ 42,  11, 252, 107],[ 56,  99,   3,  14]],
            [[ 14, 657,  33, 232],[  5,  37,   8,  22]],
            [[ 59, 104,  59,  42],[ 13, 134,  16,  10]],
            [[ 13, 313,  53,  39],[ 25,  54,  29,   7]],
            [[ 52,  41, 116,   3],[ 67, 171,   7,   2]],
            [[ 18,  56,  84,   6],[ 14, 156,  30,  24]],
            [[ 51, 293,  73, 136],[ 18,   8,  13,   6]],
            [[ 14, 167,  53, 549],[ 14,  48,   5,  39]],
            [[  3,   6, 159, 291],[  9, 149,  16,  48]],
            [[ 10,  77,  90,  43],[ 32,  80,  19,  64]],
            [[ 14,  13, 103, 200],[  7, 520,   6,  30]],
            [[ 18,  21,  54, 171],[ 41, 216,   5,  64]],
            [[ 19, 183,  16, 243],[ 41,  70,  18,  11]],
            [[ 27,  11,  68,  73],[ 12, 114,  14,  80]],
            [[  5,  93, 156,  88],[ 26, 361,   9,  42]],
            [[ 12, 533,  15,  18],[ 25, 148,  15,  21]],
            [[ 23, 124, 168, 157],[  6, 160,   5,  63]],
            [[ 40,  35, 209,  76],[ 49, 186,   4,  16]],
            [[ 19, 239,   7,  13],[ 55, 175,  20,   8]],
            [[ 14, 578,  39, 153],[ 15, 101,  14,   7]]
       ]
    if split_mode == "utkface_5_natural":
        client_samples = [
            [[ 73,  53,   3,   2,  14],[183,  25,  33, 143, 130]],
            [[248, 115, 171,  11,   4],[159,  49, 125,   6,  23]],
            [[204,  41, 153, 152,   7],[128,  91,   9,  17,  15]],
            [[  3, 230,  22, 219,  10],[ 52,  14,  36,  43,   7]],
            [[404,   6, 237,  42,  68],[122,  12,  32,  18,  10]],
            [[ 90, 220,  23,  65,   5],[ 33,  39,  38,  87,  60]],
            [[567,  49,  55,  17,  16],[163, 115,  22,   8,  16]],
            [[107, 142,  49,  16,  30],[ 64, 211,  14,  27,  14]],
            [[476,  19, 103,   3,  77],[198,  40,   3,  44,  15]],
            [[140,  23,  68,   3,  15],[165, 194,  50,  35,  43]],
            [[498, 139,  70,  54,  22],[ 10,  88,  12,  21,   5]],
            [[115,  73,  46, 195,  16],[ 55,  21,  89,  61,  12]],
            [[  6,   4, 129,  97,   9],[159, 100, 102,  21,  47]],
            [[ 70,  33,  75,  16,  34],[ 88, 125, 141,  43,  31]],
            [[108,   6,  86,  69,   8],[567,  33,  64,   3,  49]],
            [[136,   9,  42,  54,  41],[217,  22, 129, 140,  11]],
            [[144,  73,  13,  80,  43],[ 73, 110,  22,  21,  52]],
            [[165,   5,  43,  20,  10],[ 94,  65, 133, 221,  19]],
            [[ 22,  31, 102,  25,  22],[310,  40,  72, 102,  61]],
            [[ 80, 194,  11,   7,  24],[142,  86,  39, 144,  19]],
            [[206,  58, 156,  60,   7],[195,  29, 154,   8,  17]],
            [[276,  13, 141,  22,  43],[165,  13,  26,  82,  63]],
            [[155, 102,   6,   6,  62],[198, 131,  16,  71,  11]],
            [[ 99, 215,  29,  48,  14],[100,  76,  12, 143,  14]]
       ]
    if split_mode in ["fmow_1", "fmow_3"]:
        global_dist_target = np.array([
            [ 2775, 17494],
            [ 1517,  1325],
            [ 2257,  2099],
            [ 1445,  2491],
            [ 1709,  2876],
            [ 5575,  1044],
            [ 2030,  1376],
            [ 1342,  3243],
            [ 1361,  3554],
            [16209,  7695],
            [ 1504,  1955],
            [ 1911,  2329],
            [ 1258,  1311],
            [ 3057,  1050],
            [ 1022,  1842],
            [ 1337,  2202],
            [ 2266,  1885],
            [ 1507,  4247],
            [ 2877,  2095],
            [ 3474,  3116],
            [ 2740,  1704],
            [ 2062,  1019],
            [ 2334,  2061],
            [ 2275,  2132],
            [ 2246,  2708],
            [ 2015,  1644],
            [ 7354,  3257],
            [ 2749,  1076],
            [ 1215,  1683],
            [ 2116,  1409],
            [ 5724,  4567],
            [ 2241,  2562],
            [ 1630,  2276],
            [ 5798,  2468],
            [ 1654,  2374],
            [ 2490,  2340]])
        num_clients = conf['dataset_options']['num_clients']
        num_attributes = conf['dataset_options']['num_groups']
        num_classes = conf['dataset_options']['num_targets']
        alpha = 0.7
        min_samples = 2
        bin_size = 50
        client_types = [("a",20),("b",20),("d",30)]
        a_max = 6000
        if split_mode=="fmow_1":
            client_samples = generate_clients_with_global_params(global_dist_target,num_clients,num_attributes,num_classes,alpha,min_samples,seed)
        if split_mode=="fmow_3":
            client_samples = gen_clients(global_dist_target, min_samples, num_clients, num_classes, client_types, bin_size, seed, a_max)
        if split_mode=="fmow_2":
            raise NotImplementedError("Split mode had an error, removed")

    return client_samples

def generate_clients_with_global_params(global_dist_target,num_clients,num_attributes,num_classes,alpha,min_samples,seed):
    assert global_dist_target.min()>=min_samples*num_clients, "Global target not possible with min_samples guarantee"
    global_dist = global_dist_target-(num_clients*min_samples)
    client_samples = []
    for i in range(num_clients):
        seed_i = seed * i if seed is not None else None
        alphas = np.random.default_rng(seed_i).dirichlet([alpha]*(num_attributes*num_classes),1)[0]
        client_dist = list_to_matrix(alphas, num_classes, num_attributes)
        client_dist = client_dist *(num_attributes*num_classes)
        client_samples.append(list(client_dist))
    client_samples = np.array(client_samples)
    client_samples = client_samples/sum(client_samples) * global_dist
    client_samples = (np.round(client_samples)+min_samples).astype(int)
    return client_samples


# New way of client generation
class Seeder():
    def __init__(self, seed=None):
        self.seed = seed
        self.next_seed()

    def next_seed(self):
        current_seed = self.seed
        new_seed =  np.random.default_rng(self.seed).integers(0,100000,1)[0]
        self.seed = new_seed
        return current_seed

def gen_c_random(bins, n, seeder):
    
    orig_shape = bins.shape
    bins = bins.reshape(bins.size)
    taken = np.zeros_like(bins)
    bins = np.array(bins)
    taken = np.zeros_like(bins)

    for _ in range(n):
        available_indices = np.where(bins > 0)[0]
        if available_indices.size == 0:
            break  # Stop if there are no more available items to take

        chosen_index = np.random.default_rng(seeder.next_seed()).choice(available_indices)  # Randomly choose an index
        taken[chosen_index] += 1  # Increase taken count
        bins[chosen_index] -= 1  # Decrease remainder count

    return taken.reshape(orig_shape), bins.reshape(orig_shape)


def gen_c_classheavy(in_bins, bin_per_client, seeder):
    """Client that has attribute balanced samples heavily from one class"""
    client_bins = np.zeros(in_bins.shape)
    min_bins = in_bins.min(axis=1)
    if max(min_bins)>np.ceil(bin_per_client/in_bins.shape[1]):
        class_id = np.argmax(min_bins)
        client_bins[class_id] += (bin_per_client//in_bins.shape[1])
        rem_bins = in_bins - client_bins
        return client_bins, rem_bins
    if max(min_bins)>=1:
        class_id = np.argmax(min_bins)
        client_bins[class_id] += max(min_bins)
        rem_bins = in_bins - client_bins
        bins_to_fill = int(bin_per_client-(max(min_bins)*in_bins.shape[1]))
        crem, _ = gen_c_classheavy(rem_bins, bins_to_fill, seeder=seeder)
        client_bins += crem
        rem_bins = in_bins - client_bins
        return client_bins, rem_bins

    return gen_c_random(in_bins, bin_per_client, seeder=seeder)


def gen_c_attributeheavy(in_bins, bin_per_client, seeder):
    """Tries to get bins from one attribute but class balanced"""
    if in_bins.min(axis=0).max()>0:
        attr_id = np.argmax(in_bins.min(axis=0))
        x = np.zeros_like(in_bins)
        x[:,attr_id]  = in_bins[:,attr_id]
        crem, _ =  gen_c_random(x,bin_per_client, seeder=seeder)
        rem_bins = in_bins - crem
        return crem, rem_bins
    attr_id = np.argmax(in_bins.sum(axis=0))
    if in_bins[:,attr_id].sum()>bin_per_client:
        x = np.zeros_like(in_bins)
        x[:,attr_id]  = in_bins[:,attr_id]
        crem, _ =  gen_c_random(x,bin_per_client, seeder=seeder)
        rem_bins = in_bins - crem
        return crem, rem_bins
    return gen_c_random(in_bins, bin_per_client, seeder=seeder)


def gen_c_attr_balanced_more_class(in_bins, bin_per_client, seeder):
    """Tries to get bins such that the client has as many classes as possible with balanced attributes"""

    orig_shape = in_bins.shape
    num_attr = in_bins.shape[1]
    if in_bins.min(axis=1).sum()>=np.ceil(bin_per_client/num_attr):
        x = in_bins.min(axis=1)
        bin_per_attr = bin_per_client//num_attr
        cbins, _ = gen_c_random(x,bin_per_attr, seeder=seeder)
        cbins = cbins.repeat(num_attr).reshape(orig_shape)
        rem = in_bins - cbins
        bin_rem = bin_per_client-cbins.sum()
        crem, _ = gen_c_random(rem,int(bin_rem), seeder=seeder)
        cbins = cbins + crem
        rem_bins = in_bins - cbins
        return cbins, rem_bins
    # TODO: Maybe another fallback?
    return gen_c_random(in_bins, int(bin_per_client), seeder=seeder)

def gen_c_diagonal(in_bins, bin_per_client,start_attr=0, seeder=None):
    """For each class tries to pick different attribute"""
    orig_shape = in_bins.shape
    num_attr = in_bins.shape[1]
    cbins = np.zeros_like(in_bins)
    if in_bins.min(axis=1).sum()>=bin_per_client:
        x = in_bins.min(axis=1)
        cline, _ = gen_c_random(x,bin_per_client, seeder=seeder)
        for i in range(len(cline)):
            shift = (i+start_attr) % num_attr
            cbins[i,shift] = cline[i]
        rem = in_bins - cbins
        bin_rem = bin_per_client-cline.sum()
        crem, _ = gen_c_random(rem,int(bin_rem), seeder=seeder)
        cbins = cbins + crem
        rem_bins = in_bins - cbins
        return cbins, rem_bins
    # TODO: Maybe another fallback?
    return gen_c_random(in_bins, int(bin_per_client), seeder=seeder)


def gen_clients(global_dist_target, min_samples, num_clients, num_classes, client_types=[("r",1)], bin_size=50, seed=None, a_max=None):
    seeder = Seeder(seed)
    global_dist_target = np.clip(global_dist_target, a_min=0, a_max=a_max)
    global_dist_target = global_dist_target[:num_classes]
    global_dist_target = global_dist_target - (num_clients * min_samples)
    global_dist_target = global_dist_target // bin_size
    bin_per_client = int(global_dist_target.sum()/num_clients)
    client_samples = []
    for method_type, num_next in client_types:
        for i in range(num_next):
            if len(client_samples)==num_clients:
                break
            if method_type=="r":
                client_bins, global_dist_target = gen_c_diagonal(global_dist_target, bin_per_client, seeder=seeder)
            elif method_type=="d":
                client_bins, global_dist_target = gen_c_diagonal(global_dist_target, bin_per_client, start_attr=i, seeder=seeder)
            elif method_type=="c":
                client_bins, global_dist_target = gen_c_classheavy(global_dist_target, bin_per_client, seeder=seeder)
            elif method_type=="a":
                client_bins, global_dist_target = gen_c_attributeheavy(global_dist_target, bin_per_client, seeder=seeder)
            elif method_type=="b":
                client_bins, global_dist_target = gen_c_attr_balanced_more_class(global_dist_target, bin_per_client, seeder=seeder)
            else:
                raise NotImplementedError("Unrecognized client type")
            client_samples.append(client_bins)
    if len(client_samples)<num_clients:
        for i in range(len(client_samples),num_clients):
            client_bins, global_dist_target = gen_c_diagonal(global_dist_target, bin_per_client, seeder=seeder)
            client_samples.append(client_bins)
    client_samples = np.array(client_samples)
    client_samples = client_samples * bin_size + min_samples
    client_samples = client_samples.astype(int)
    return client_samples