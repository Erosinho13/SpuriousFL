# Algorithms implemented in https://github.com/YyzHarry/SubpopBench/blob/main/subpopbench/learning/algorithms.py

import os
from src.datasets.dataset_utils import get_metadata
from src.models.afed_generator import AFedGenerator
import torch
import numpy as np
from src.utils import get_device
import copy
import inspect

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
        self.hparams = conf["client_opt"]
        if "dataset_options" in conf.keys():
            dopt = conf["dataset_options"]
            if "num_targets" in dopt.keys():
                self.num_classes = dopt["num_targets"]
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


def is_two_stage_optimizer(conf):
    if "client_opt" in conf.keys():
        copt = conf["client_opt"]
        if "subpop_optimizer" in copt.keys():
            spopt = copt["subpop_optimizer"]
            if "CRT" in spopt or "DFR" in spopt or "FEx" in spopt:
                return True
            return False
    raise KeyError("Missing conf.client_opt.subpop_optimizer")


def get_subpop_optimizer(model, data, conf={}):
    if "client_opt" in conf.keys():
        copt = conf["client_opt"]
        if "subpop_optimizer" in copt.keys():
            if copt["subpop_optimizer"] == "ERM":
                return ERM(model, conf)
            elif copt["subpop_optimizer"] == "GroupDRO":
                return GroupDRO(model, conf)
            elif copt["subpop_optimizer"] == "ReSample":
                return ReSample(model, conf)
            elif copt["subpop_optimizer"] == "ReWeight":
                metadata = get_metadata(data, conf)
                return ReWeight(model, conf, metadata)
            elif copt["subpop_optimizer"] == "SqrtReWeight":
                metadata = get_metadata(data, conf)
                return SqrtReWeight(model, conf, metadata)
            elif copt["subpop_optimizer"] == "CBLoss":
                metadata = get_metadata(data, conf)
                return CBLoss(model, conf, metadata)
            elif copt["subpop_optimizer"] == "Focal":
                return Focal(model, conf)
            elif copt["subpop_optimizer"] == "CRT":
                return CRT(model, conf)
            elif copt["subpop_optimizer"]== "ReWeightCRT":
                metadata = get_metadata(data, conf)
                return ReWeightCRT(model, conf, metadata)
            elif copt["subpop_optimizer"] == "DFR":
                return DFR(model, conf)
            elif copt["subpop_optimizer"] == "FEx":
                return FEx(model, conf)
            elif copt["subpop_optimizer"] == "FExCRT":
                return FExCRT(model, conf)
            elif copt["subpop_optimizer"] == "LfF":
                return LfF(model, conf)
            elif copt["subpop_optimizer"] == "Prox":
                return Prox(model, conf)
            else:
                raise NotImplementedError("Subpop optimizer not recognized")
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
        signature = inspect.signature(opt.__init__)
        if 'momentum' in copt.keys() and 'momentum' in signature.parameters:
            return opt(params, lr=lr, momentum=copt['momentum'])
        return opt(params, lr=lr)
    return torch.optim.SGD(params, lr=0.001)


def get_sample_weights(dataset, conf):
    """ReSample weights for DataLoader.sampler"""
    if "client_opt" in conf.keys():
        copt = conf["client_opt"]
        if "subpop_optimizer" in copt.keys():
            if copt["subpop_optimizer"] in ["ReSample", "ReWeight"]:
                dataset.update_metadata()
            if copt["subpop_optimizer"] == "ReSample":
                # if attribute not available, groups degenerate to classes
                train_weights = np.asarray(dataset.weights_g)
                train_weights /= np.sum(train_weights)
                return train_weights
    return None


class GeneralizedCrossEntropyLoss(torch.nn.modules.loss._Loss):
    """Generalized Cross Entropy Loss from https://proceedings.neurips.cc/paper_files/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf"""
    def __init__(self, q=0.7,reduction='mean'):
        """
        Custom CrossEntropyLoss implementation.

        Args:
            q: a hyperparameter that controls the degree of amplification
            reduction (str, optional): Specifies the reduction to apply to the output.
                                       Must be one of 'none', 'mean', or 'sum'. Default: 'mean'
        """
        self.q = q
        super(GeneralizedCrossEntropyLoss, self).__init__(reduction=reduction)
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Invalid reduction type: {reduction}. Choose from 'none', 'mean', or 'sum'.")

    def forward(self, logits, target):
        # Step 1: Compute the softmax
        softmax_probs = torch.nn.functional.softmax(logits, dim=1)

        # Step 2: Select the correct class probabilities
        correct_class_probs = softmax_probs[range(logits.shape[0]), target]

        # Compute loss
        loss = (1-(correct_class_probs**self.q))/self.q

        # Step 4: Apply reduction ('none', 'mean', or 'sum')
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # self.reduction == 'none'
            return loss


class CustomCrossEntropyLoss(torch.nn.modules.loss._Loss):
    """Local implementation of CrossEntropy loss in Pytorch"""
    def __init__(self, reduction='mean'):
        """
        Custom CrossEntropyLoss implementation.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the output.
                                       Must be one of 'none', 'mean', or 'sum'. Default: 'mean'
        """
        super(CustomCrossEntropyLoss, self).__init__(reduction=reduction)
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Invalid reduction type: {reduction}. Choose from 'none', 'mean', or 'sum'.")

    def forward(self, logits, target):
        """
        Forward pass of the custom cross-entropy loss.

        Args:
            logits (Tensor): The input tensor of shape (batch_size, num_classes) containing raw, unnormalized scores.
            target (Tensor): The target tensor of shape (batch_size,) containing class indices.

        Returns:
            Tensor: The computed cross-entropy loss.
        """
        # Step 1: Compute the softmax
        softmax_probs = torch.nn.functional.softmax(logits, dim=1)

        # Step 2: Select the correct class probabilities
        correct_class_probs = softmax_probs[range(logits.shape[0]), target]

        # Step 3: Compute the negative log-likelihood
        loss = -torch.log(correct_class_probs)

        # Step 4: Apply reduction ('none', 'mean', or 'sum')
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # self.reduction == 'none'
            return loss


def get_loss(mode="cross_entropy", conf={}):
    if "client_opt" in conf.keys():
        if "loss_function" in conf["client_opt"].keys():
            mode = conf["client_opt"]["loss_function"]
    if mode == "generalized_cross_entropy":
        if "generalized_cross_entropy_q" in conf["client_opt"]:
            q = conf["client_opt"]["generalized_cross_entropy_q"]
            return GeneralizedCrossEntropyLoss(q=q, reduction="none")
        return GeneralizedCrossEntropyLoss(reduction="none")
    if mode == "custom_cross_entropy":
        return CustomCrossEntropyLoss(reduction="none")
    if mode == "cross_entropy":
        return torch.nn.CrossEntropyLoss(reduction="none")
    raise NotImplementedError("Unrecognized loss function: ", mode)


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
        self.loss = get_loss(conf=self.conf)
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


class GroupDRO(ERM):
    """
    Group DRO minimizes the error at the worst group [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, model, conf):
        super(GroupDRO, self).__init__(
            model, conf)
        self.register_buffer(
            "q", torch.ones(self.num_classes * self.num_attributes).to(get_device(conf)))

    def _compute_loss(self, i, pred, y, a, step):
        losses = self.loss(pred, y)

        for idx_g, idx_samples in self.return_groups(y, a):
            self.q[idx_g] *= (float(self.hparams["groupdro_eta"]) * losses[idx_samples].mean()).exp().item()

        self.q /= self.q.sum()

        loss_value = 0
        for idx_g, idx_samples in self.return_groups(y, a):
            loss_value += self.q[idx_g] * losses[idx_samples].mean()

        return loss_value

class ReSample(ERM):
    """Naive resample, with no changes to ERM, but enable balanced sampling in DataLoader. See: get_sample_weights()"""


class ReWeight(ERM):
    """Naive inverse re-weighting"""
    def __init__(self, model, conf, metadata):
        super(ReWeight, self).__init__(
            model, conf)
        assert len(metadata['group_sizes']) == self.num_classes * self.num_attributes
        grp_sizes = [x if x else np.inf for x in metadata['group_sizes']]
        per_grp_weights = 1 / np.array(grp_sizes)
        per_grp_weights = per_grp_weights / np.sum(per_grp_weights) * len(grp_sizes)
        self.weights_per_grp = torch.FloatTensor(per_grp_weights)

    def _compute_loss(self, i, pred, y, a, step):
        losses = self.loss(pred, y)

        all_g = y * self.num_attributes + a
        loss_value = (self.weights_per_grp.type_as(losses)[all_g] * losses).mean()

        return loss_value


class SqrtReWeight(ReWeight):
    """Square-root inverse re-weighting"""
    def __init__(self, model, conf, metadata):
        super(SqrtReWeight, self).__init__(
            model, conf, metadata)
        grp_sizes = [x if x else np.inf for x in metadata['group_sizes']]
        per_grp_weights = 1 / np.sqrt(np.array(grp_sizes))
        per_grp_weights = per_grp_weights / np.sum(per_grp_weights) * len(grp_sizes)
        self.weights_per_grp = torch.FloatTensor(per_grp_weights)


class CBLoss(ReWeight):
    """Class-balanced loss, https://arxiv.org/pdf/1901.05555.pdf"""
    def __init__(self, model, conf, metadata):
        super(CBLoss, self).__init__(
            model, conf, metadata)

        grp_sizes = [x if x else np.inf for x in metadata['group_sizes']]
        effective_num = 1. - np.power(self.hparams["cbloss_beta"], grp_sizes)
        effective_num = np.array(effective_num)
        effective_num[effective_num == 1] = np.inf
        per_grp_weights = (1. - self.hparams["cbloss_beta"]) / effective_num
        per_grp_weights = per_grp_weights / np.sum(per_grp_weights) * len(grp_sizes)
        self.weights_per_grp = torch.FloatTensor(per_grp_weights)


class Focal(ERM):
    """Focal loss, https://arxiv.org/abs/1708.02002"""
    def __init__(self, model, conf):
        super(Focal, self).__init__(
            model, conf)

    @staticmethod
    def focal_loss(input_values, gamma):
        p = torch.exp(-input_values)
        loss = (1 - p) ** gamma * input_values
        return loss.mean()

    def _compute_loss(self, i, pred, y, a, step):
        losses = self.loss(pred, y)
        return self.focal_loss(losses, self.hparams["focal_gamma"])


class CRT(ERM):
    """Classifier re-training with balanced sampling during the second learning stage"""
    def __init__(self, model, conf):
        super(CRT, self).__init__(model, conf)
        # fix stage 1 trained featurizer
        for name, param in self.featurizer.named_parameters():
            param.requires_grad = False
        # only optimize the classifier
        self.optimizer = get_base_optimizer(self.classifier.parameters(), conf)


class ReWeightCRT(ReWeight, CRT):
    """Classifier re-training with balanced re-weighting during the second learning stage"""



class DFR(CRT):
    """
    Classifier re-training with sub-sampled, group-balanced, held-out(validation) data and l1 regularization.
    Note that when attribute is unavailable in validation data, group-balanced reduces to class-balanced.
    https://openreview.net/pdf?id=Zb6c8A-Fghk
    """
    def _compute_loss(self, i, pred, y, a, step):
        return self.loss(pred, y).mean() + self.hparams['dfr_reg'] * torch.norm(self.classifier.weight, 1)


class FEx(ERM):
    """
    Forgetting examples: re-training with sub-sampled dataset. Dataset samples detected by training
    shallow model and tracking forgetting events.
    https://arxiv.org/abs/1911.03861
    """


class FExCRT(CRT):
    """Forgetting examples but with classifier re-training"""

class LfF(Algorithm):
    """
    Learning from Failure (LfF) [https://arxiv.org/pdf/2007.02561.pdf]
    """
    def __init__(self, model, conf):
        super().__init__(
            model, conf)

        pred_model_conf = copy.deepcopy(conf)
        pred_model_conf["client_opt"]["loss_function"] = "cross_entropy"
        self.pred_model = ERM(model, pred_model_conf)
        self.biased_network = copy.deepcopy(model)
        self._init_model()

    def _init_model(self):
        self.pred_model._init_model()
        
        self.optimizer_b = get_base_optimizer(self.network.parameters(), self.conf)
        if "generalized_cross_entropy_q" in self.hparams:
            gce_q = self.hparams["generalized_cross_entropy_q"]
        else:
            gce_q = 0.7
        self.gce_loss = get_loss(conf={"client_opt":{"loss_function":"generalized_cross_entropy",
                                                     "generalized_cross_entropy_q":gce_q}})
        self.lr_scheduler = None

    def _compute_loss(self, i, pred, y, a, step):
        pred_logits, biased_logits = pred
        loss_gce = self.gce_loss(biased_logits, y)
        ce_b = torch.nn.functional.cross_entropy(biased_logits, y, reduction='none')
        ce_d = torch.nn.functional.cross_entropy(pred_logits, y, reduction='none')
        weights = (ce_b/(ce_b + ce_d + 1e-8)).detach()
        loss_pred = (ce_d * weights).mean()
        loss = loss_pred.mean() + loss_gce.mean()
        return loss, loss_pred, loss_gce


    def update(self, minibatch, step):
        all_i, all_x, all_y, all_a = minibatch   
        pred_logits = self.pred_model.predict(all_x) 
        biased_logits = self.biased_network(all_x)

        self.optimizer_b.zero_grad()
        self.pred_model.optimizer.zero_grad()

        loss, loss_pred, loss_gce = self._compute_loss(all_i, (pred_logits, biased_logits), all_y, all_a, step)
        loss.backward()

        self.optimizer_b.step()
        self.pred_model.optimizer.step()

        correct = (torch.max(pred_logits.data, 1)[1] == all_y).sum().item()
        return {'loss': loss.item(), 'loss_pred': loss_pred.mean().item(), 'loss_gce': loss_gce.mean().item(), "correct":correct}

    def return_feats(self, x):
        return self.pred_model.featurizer(x)

    def predict(self, x):
        return self.pred_model.predict(x)


class Prox(ERM):
    """ERM with Proximal loss (from fedprox: https://flower.ai/docs/baselines/fedprox.html)"""
    def __init__(self, model, conf):
        super(ERM, self).__init__(
            model, conf)

        self.featurizer = self.network.featurizer
        self.classifier = self.network.classifier
        self._init_model()

    def _init_model(self):
        self.orig_model_params = copy.deepcopy(self.network).parameters()
        self.proximal_mu = self.hparams["proximal_mu"]
        self.optimizer = get_base_optimizer(self.network.parameters(), self.conf)
        self.loss = get_loss(conf=self.conf)
        self.lr_scheduler = None


    def _compute_loss(self, i, pred, y, a, step):
        orig_loss =  self.loss(pred, y).mean()
        if self.proximal_mu == 0.0:
            return orig_loss
        proximal_term = 0.0
        for local_weights, global_weights in zip(self.network.parameters(), self.orig_model_params):
            proximal_term += torch.square((local_weights - global_weights).norm(2))
        loss = orig_loss + (self.proximal_mu / 2) * proximal_term
        return loss


class AFed(Algorithm):
    """AFed uses a generator (obtained from the server) to optimize training for global data distribution.
    AFed paper: https://arxiv.org/pdf/2501.02732"""
    def __init__(self, model, conf):
        super().__init__(
            model, conf)
        #!TODO: Load generator from probably file
        model_path = os.path.join(
            "checkpoints",
            self.conf["exp_id"],
            "afed_generator"
        )
        generator = AFedGenerator()
        generator.load_state_dict(torch.load(os.path.join(model_path, "torchmodel.pt")))

        self.featurizer = self.network.featurizer
        self.classifier = self.network.classifier
        attr_num = conf["dataset_options"]["num_groups"]
        device = get_device(conf)
        self.a_classifier = torch.nn.Linear(self.network.classifier.in_features, attr_num).to(device)
        self._init_model()

    def _init_model(self):
        lr = self.conf["client_opt"]["learning_rate"]
        self.optimizer = {
            'featurizer': torch.optim.Adam(self.featurizer.parameters(), lr=lr),
            'classifier': torch.optim.SGD(self.classifier.parameters(), lr=lr),
            'a_classifier': torch.optim.SGD(self.a_classifier.parameters(), lr=lr)
        }
        self.loss = {
            'classifier': get_loss(conf=self.conf),
            'a_classifier': get_loss(conf=self.conf)
        }
        self.lr_scheduler = None

    def _compute_loss(self, i, pred, y, a, step):
        # Not used
        return self.loss['classifier'](pred, y).mean()

    def update(self, minibatch, step):
        all_i, all_x, all_y, all_a = minibatch
        self.optimizer['featurizer'].zero_grad()
        self.optimizer['classifier'].zero_grad()
        self.optimizer['a_classifier'].zero_grad()
        all_feat = self.return_feats(all_x)
        y_scores = self.classifier(all_feat)
        y_loss_true = self.loss['classifier'](y_scores.view(-1), all_y)
        y_loss_value = y_loss_true.item()
        a_scores = self.a_classifier(all_feat.detach())
        a_loss = self.loss['a_classifier'](a_scores, all_a).mean()
        a_loss.backward()
        a_loss_value = a_loss.item()
        self.optimizer['a_classifier'].step()

        # Generator G:
        alpha = 1 
        gamma = np.random.beta(alpha, alpha)
        true_feat_0 = self.return_feats(all_x)
        z_attr, _ = gen_z_attr(1-all_a, z_dim=self.conf["client_opt"]["AFed_generator_noise_dim"] + 2)
        fake_feat_1 = self.generator(z_attr.to(get_device(self.conf)))
        mix_feat = gamma * true_feat_0 + (1-gamma) * fake_feat_1

        mix_feat = mix_feat.requires_grad_(True)
        pred = self.classifier(mix_feat).sum()
        grad = torch.autograd.grad(outputs=pred, inputs=mix_feat, create_graph=True)[0].view(mix_feat.size(0), -1)
        delta_x = (true_feat_0 - fake_feat_1).view(mix_feat.size(0), -1)
        grad_inn = (grad * delta_x).sum(1).view(-1)
        loss_grad = torch.abs(grad_inn.mean())
        y_loss_true += self.conf["client_opt"]["AFed_lam"] * loss_grad

        y_loss_true.backward()
        self.optimizer['featurizer'].step()
        self.optimizer['classifier'].step()


        correct = (torch.max(y_scores.data, 1)[1] == all_y).sum().item()
        return {'loss': y_loss_value, "correct":correct, 'a_loss': a_loss_value}

    def return_feats(self, x):
        return self.featurizer(x)

    def predict(self, x):
        return self.network(x)


def gen_z_attr(a, num_classes=2, z_dim=34):
    batch_size = a.shape[0]
    eps = torch.rand(batch_size, z_dim)
    eps = torch.FloatTensor(eps)
    # https://arxiv.org/abs/1809.03627
    one_hot_attr = torch.zeros(batch_size, num_classes)
    one_hot_attr.scatter_(1, a.cpu().type(torch.int64).view(-1, 1), 1)
    z_y_attr = torch.cat((eps, one_hot_attr), dim=1)

    return z_y_attr, eps
    
