# Algorithms implemented in https://github.com/YyzHarry/SubpopBench/blob/main/subpopbench/learning/algorithms.py

import torch

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a subgroup robustness algorithm.
    Subclasses should implement the following:
    - _init_model()
    - _compute_loss()
    - update()
    - return_feats()
    - predict()
    """
    def __init__(self, model, conf):
        super(Algorithm, self).__init__()
        self.conf = conf
        if "dataset_options" in conf.keys():
            dopt = conf["dataset_options"]
            if "num_targets" in dopt.keys():
                self.num_labels = dopt["num_targets"]
            if "num_groups" in dopt.keys():
                self.num_attributes =  dopt["num_groups"]
        self.network = model

    def _init_model(self):
        raise NotImplementedError

    def _compute_loss(self, i, pred, y, a, step):
        raise NotImplementedError

    def update(self, minibatch, step):
        """Perform one update step."""
        raise NotImplementedError

    def return_feats(self, x):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def return_groups(self, y, a):
        """Given a list of (y, a) tuples, return indexes of samples belonging to each subgroup"""
        idx_g, idx_samples = [], []
        all_g = y * self.num_attributes + a

        for g in all_g.unique():
            idx_g.append(g)
            idx_samples.append(all_g == g)

        return zip(idx_g, idx_samples)

    @staticmethod
    def return_attributes(all_a):
        """Given a list of attributes, return indexes of samples belonging to each attribute"""
        idx_a, idx_samples = [], []

        for a in all_a.unique():
            idx_a.append(a)
            idx_samples.append(all_a == a)

        return zip(idx_a, idx_samples)


def get_subpop_optimizer(model, conf={}):
    if "client_opt" in conf.keys():
        copt = conf["client_opt"]
        if "subpop_optimizer" in copt.keys():
            if copt["subpop_optimizer"] == "ERM":
                return ERM(model, conf)
    return ERM(model, conf)


def get_base_optimizer(params, conf={}):
    if "client_opt" in conf.keys():
        copt = conf["client_opt"]
        if "base_optimizer" in copt.keys():
            if copt["base_optimizer"] == "SGD":
                opt = torch.optim.SGD
            elif copt["base_optimizer"] == "Adam":
                opt = torch.optim.Adam
            else:
                raise NotImplementedError(f"optimizer not recognized {copt['base_optimizer']}")
        if "learning_rate" in copt:
            lr = float(copt["learning_rate"])
        else:
            lr = 0.001
        return opt(params, lr=lr)
    return torch.optim.SGD(params, lr=0.001)


def get_loss(conf={}):
    return torch.nn.functional.cross_entropy


class ERM(Algorithm):
    """Empirical Risk Minimization (ERM)"""
    def __init__(self, model, conf):
        super(ERM, self).__init__(
            model, conf)

        self.featurizer = self.network.featurizer
        self.classifier = self.network.classifier
        self._init_model()

    def _init_model(self):
        self.optimizer = get_base_optimizer(self.network.parameters(), self.conf)
        self.loss = get_loss(self.conf)
        self.lr_scheduler = None


    def _compute_loss(self, i, pred, y, a, step):
        return self.loss(pred, y).mean()

    def update(self, minibatch, step):
        all_i, all_x, all_y, all_a = minibatch
        all_pred = self.predict(all_x)
        loss = self._compute_loss(all_i, all_pred, all_y, all_a, step)

        self.optimizer.zero_grad()
        loss.backward()
        # if self.clip_grad:
        #    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        #if self.lr_scheduler is not None:
        #    self.lr_scheduler.step()

        #if self.data_type == "text":
        #    self.network.zero_grad()

        correct = (torch.max(all_pred.data, 1)[1] == all_y).sum().item()
        return {'loss': loss.item(), "correct":correct}

    def return_feats(self, x):
        return self.featurizer(x)

    def predict(self, x):
        return self.network(x)
