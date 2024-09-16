from src.optimizers.subpopbench import GroupDRO
import numpy as np

def store_opt_params(opt, out_dict):
    """Save important params of the opt into a dict"""
    if isinstance(opt, GroupDRO):
        for i, q in enumerate(opt.q):
            out_dict["GroupDRO_q_"+str(i)] = q.item()
    return out_dict


def init_shared_opt_params(conf):
    """Create dict with init value of shared params"""
    out_dict = {}
    if conf["client_opt"]["subpop_optimizer"] == "GroupDRO":
        n = conf["dataset_options"]["num_targets"] * conf["dataset_options"]["num_groups"]
        for i in range(n):
            out_dict["GroupDRO_q_"+str(i)] = 1.0
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
            avg = np.average([c_res[k] for c_res in metric_list])
            shared_opt_params[k] = avg
    return shared_opt_params