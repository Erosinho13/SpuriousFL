from src.datasets.dataset_utils import count_groups
from src.optimizers.subpopbench import GroupDRO
import numpy as np
from src import corr

def store_opt_params(opt, out_dict, data, conf):
    """Save important params of the opt into a dict"""
    if isinstance(opt, GroupDRO):
        for i, q in enumerate(opt.q):
            out_dict["GroupDRO_q_"+str(i)] = q.item()
    if conf["server_opt"]["weight_clients"].startswith("server_post_groupweights"):
        metadata = count_groups(data, 
                                num_attributes=conf["dataset_options"]["num_groups"],
                                num_labels=conf["dataset_options"]["num_targets"])

        for i, v in enumerate(metadata["group_sizes"]):
            out_dict["groupsize_"+str(i)] = v
    if conf["server_opt"]["weight_clients"].startswith("server_post_triplets"):
        metadata = count_groups(data, 
                                num_attributes=conf["dataset_options"]["num_groups"],
                                num_labels=conf["dataset_options"]["num_targets"])
        N = np.resize(np.array(metadata["group_sizes"]),
                      (conf["dataset_options"]["num_targets"],conf["dataset_options"]["num_groups"]))
        out_dict["SC"] = corr.SC(N)
        out_dict["AI"] = corr.AI(N)
        out_dict["CI"] = corr.CI(N)
    return out_dict


def init_shared_opt_params(conf):
    """Create dict with init value of shared params"""
    out_dict = {}
    if conf["client_opt"]["subpop_optimizer"] == "GroupDRO":
        n = conf["dataset_options"]["num_targets"] * conf["dataset_options"]["num_groups"]
        for i in range(n):
            out_dict["GroupDRO_q_"+str(i)] = 1.0
    if conf["server_opt"]["weight_clients"].startswith("server_post_groupweights"):
        n = conf["dataset_options"]["num_targets"] * conf["dataset_options"]["num_groups"]
        for i in range(n):
            out_dict["groupsize_"+str(i)] = 0
    return out_dict


def update_opt_with_shared_params(opt, param_dict):
    """Update client opt params with values from server"""
    if isinstance(opt, GroupDRO):
        for i in range(len(opt.q)):
            v = param_dict["GroupDRO_q_"+str(i)]
            opt.q[i] = v


def aggregate_metrics(shared_opt_params, metric_list):
    """Recieves flower client metric list and stores it in shared optimizer params dict for server"""
    for k in shared_opt_params.keys():
        if k in metric_list[0].keys():
            if k.startswith("GroupDRO_q_"):
                avg = np.average([c_res[k] for c_res in metric_list])
                shared_opt_params[k] = avg
            elif k.startswith("groupsize_"):
                total = float(np.sum([c_res[k] for c_res in metric_list]))
                shared_opt_params[k] = total
            else:
                print("Warning: unhandled shared params")

    return shared_opt_params