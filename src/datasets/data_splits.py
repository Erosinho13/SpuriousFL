from typing import List
import numpy as np

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

def split_mode_to_matrix(split_mode:str)-> List:
    """Named split modes to N matrix list"""
    client_samples = None
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
    if split_mode=="spawrious4":
        client_samples = []
        client_samples += [[[20, 20], [20, 20], [5, 5], [5, 5]]] * 2  # CI
        client_samples += [[[5, 5], [5, 5], [20, 20], [20, 20]]] * 2  # CI
        client_samples += [[[20, 5], [20, 5], [20, 5], [20, 5]]] * 2  # AI
        client_samples += [[[5, 20], [5, 20], [5, 20], [5, 20]]] * 2  # AI
        client_samples += [[[5, 20], [5, 20], [20, 5], [20, 5]]] *1 # SC
        client_samples += [[[119, 5], [119, 5], [5, 119], [5, 119]]] * 7  # SC
        client_samples += [[[118, 5], [118, 5], [5, 118], [5, 118]]] * 9  # SC

    return client_samples

