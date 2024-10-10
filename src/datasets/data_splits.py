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
    return client_samples