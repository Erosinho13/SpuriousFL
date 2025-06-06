import os
from typing import Callable, OrderedDict
import numpy as np
from src.models import model_utils
from src.models.afed_generator import AFedGenerator
from src.optimizers.subpopbench import gen_z_attr, get_base_optimizer, get_loss, get_subpop_optimizer
from src.utils import get_cpu, get_device, get_input_shape, np_to_tensor
import torch
from sklearn.metrics import average_precision_score


class History:
    def __init__(self):
        self.history = {"loss": [], "accuracy": []}

    def __str__(self):
        return str(self.history)


def fit(model, data, conf, validation_data=None, verbose=0, opt=None):
    model.train()  # switch to training mode
    history = History()

    if opt is None:
        opt = get_subpop_optimizer(model, data.dataset, conf)

    num_steps = count_steps = -1
    if 'num_steps' in conf["client_opt"]:
            
        if isinstance(conf["client_opt"]["num_steps"], int) and conf["client_opt"]['num_steps'] != -1:
            num_steps = conf["client_opt"]['num_steps']
            count_steps = 0
            conf["client_opt"]["epochs"] = 1000000

    for epoch in range(conf["client_opt"]["epochs"]):

        correct, total, epoch_loss = 0, 0, 0.0

        for i, (indeces, images, (labels, groups)) in enumerate(data):

            indeces = indeces.to(get_device(conf))
            images, labels = images.to(get_device(conf)), labels.to(get_device(conf))
            groups = groups.to(get_device(conf))

            opt_out = opt.update((indeces, images, labels, groups), 1)
            loss = opt_out["loss"]
            minibatch_correct = opt_out["correct"]

            # Metrics
            correct += minibatch_correct
            epoch_loss += loss * images.size(0)
            total += labels.size(0)
            del loss

            if num_steps != -1:
                count_steps += 1
                if count_steps >= num_steps:
                    break

        epoch_loss /= len(data.dataset)
        epoch_acc = correct / total

        if validation_data is not None:
            model.eval()  # validation
            with torch.no_grad():
                correct, total, val_loss = 0, 0, 0.0
                for indeces, images, (labels, groups) in validation_data:
                    indeces = indeces.to(get_device(conf))
                    groups = groups.to(get_device(conf))
                    images, labels = images.to(get_device(conf)), labels.to(get_device(conf))
                    opt_out = opt.update((indeces, images, labels, groups), 1)
                    loss = opt_out["loss"]
                    minibatch_correct = opt_out["correct"]
                    correct += minibatch_correct
                    val_loss += loss * images.size(0)
                    total += labels.size(0)
                    del loss
                val_loss /= len(validation_data.dataset)
                val_acc = correct / total
            model.train()  # switch to training mode

        if verbose > 0:
            v_string = ''
            if validation_data is not None:
                v_string = f" val_loss:{val_loss}, val_acc:{val_acc}"
            print(f"Epoch {epoch + 1}: loss:{epoch_loss:.4f}, acc:{epoch_acc:.4f}" + v_string)

        history.history["loss"].append(epoch_loss)
        history.history["accuracy"].append(epoch_acc)

        if num_steps != -1:
            if count_steps >= num_steps:
                break

    return history


def collect_lists(indeces, images, labels, groups, predicted, outdict):
    if "pred_labels" not in outdict:
        outdict["pred_labels"] = {}
    if "orig_labels" not in outdict:
        outdict["orig_labels"] = {}
    if "attributes" not in outdict:
        outdict["attributes"] = {}
    for i, l, g, v in zip(indeces.detach().to('cpu').numpy(), labels.detach().to('cpu').numpy(), groups.detach().to('cpu').numpy(), predicted.detach().to('cpu').numpy()):
            outdict["pred_labels"][i] = int(v)
            outdict["orig_labels"][i] = int(l)
            outdict["attributes"][i] = int(g)


#!TODO update with subpopbench optims
def evaluate(model, data, conf, verbose=0,
             extra_eval_fn: Callable[[
                 torch.Tensor, # indeces
                 torch.Tensor, # images
                 torch.Tensor, # labels
                 torch.Tensor, # groups
                 torch.Tensor, # predicted
                 dict], None]=None): # outdict
    if extra_eval_fn:
        extra_ret = {}
    model.eval()
    loss_fn = get_loss(conf=conf)
    correct, total, loss = 0, 0, 0.0
    label_group_correct, label_group_total = {}, {}

    with torch.no_grad():
        for indeces, images, (labels, groups) in data:
            indeces, images, labels, groups = indeces.to(get_device(conf)), images.to(get_device(conf)), labels.to(get_device(conf)), groups.to(
                get_device(conf))
            outputs = model(images)
            loss += loss_fn(outputs, labels).mean().item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if extra_eval_fn:
                extra_eval_fn(indeces, images, labels, groups, predicted, extra_ret)

            unique_pairs = torch.unique(torch.stack((labels, groups), dim=1), dim=0)
            for label, group in unique_pairs:
                mask = (labels == label) & (groups == group)
                pair_labels = labels[mask]
                pair_predictions = predicted[mask]
                correct_count = (pair_predictions == pair_labels).sum().item()
                total_count = mask.sum().item()

                pair_key = (label.item(), group.item())
                if pair_key not in label_group_total:
                    label_group_total[pair_key] = 0
                    label_group_correct[pair_key] = 0

                label_group_total[pair_key] += total_count
                label_group_correct[pair_key] += correct_count
    loss /= len(data.dataset)
    accuracy = 100 * correct / total
    group_accuracies = {("y" + str(pair[0]) + "g" + str(pair[1])): 100* label_group_correct[pair] / label_group_total[pair]
                        for pair in label_group_total}
    worst_acc = min(group_accuracies.values())
    group_accuracies["worst_group"] = worst_acc
    if extra_eval_fn:
        return loss, accuracy, group_accuracies, extra_ret
    return loss, accuracy, group_accuracies


def predict_numpy(model, X, verbose=0):
    model.eval()
    model.to(get_cpu())
    with torch.no_grad():
        images = np_to_tensor(X)
        # images = batch.to(get_cpu())
        outputs = model(images)
        return outputs


def predict_dataloader(model, dataloader, conf, apply_softmax=True):
    total_outputs = []
    labels_all = None
    model.eval()  # Optional when not using Model Specific layer
    model.to(get_device(conf))
    for images, labels in dataloader:
        l = labels.detach().numpy()
        if labels_all is None:
            labels_all = l
        else:
            labels_all = np.concatenate((labels_all, l))
        images, labels = images.to(get_device(conf)), labels.to(get_device(conf))
        outputs = model(images)
        if apply_softmax:
            outputs = torch.nn.functional.softmax(torch.Tensor(outputs), -1)
        outputs = outputs.to('cpu').detach().numpy()
        total_outputs.extend(outputs)
    return np.array(total_outputs), np.array(labels_all)


#!TODO update with subpopbench optims
def oort_stat(model, data, conf): # outdict
    stat = {}
    model.eval()
    loss_fn = get_loss(conf=conf)
    util = 0.0
    with torch.no_grad():
        for indeces, images, (labels, groups) in data:
            indeces, images, labels, groups = indeces.to(get_device(conf)), images.to(get_device(conf)), labels.to(get_device(conf)), groups.to(
                get_device(conf))
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            util += (loss**2).sum().item()

    util /= len(data.dataset)
    util = np.sqrt(util) * len(data.dataset)

    stat["oort_util"] = util
    return stat


def get_network_embeddings(model, conf, mean=0.0, std=1.0):
    """Network embeddings by noise.
    Inspired by: https://doi.org/10.1145/3638052
    more like: https://arxiv.org/pdf/2211.13975"""
    input_shape = (conf["client_opt"]["batch_size"], *get_input_shape(conf))
    images = np.random.default_rng(conf["seed"]).normal(mean, std, input_shape)
    images = torch.from_numpy(images)
    images = images.to(get_device(conf), dtype=torch.float)
    stat = {}
    model.eval()
    final_embeddings = [0.0] * conf["dataset_options"]["num_targets"]
    with torch.no_grad():
        outputs = model(images)
        final_embeddings = outputs.mean(dim=0).detach().cpu().numpy()

    for i,v in enumerate(final_embeddings):
        stat["netemb_"+str(i)] = float(v)
    return stat

# HCSFed https://arxiv.org/pdf/2208.05135

def get_param_values(state_dict: OrderedDict) -> np.ndarray:
    """
    :param state_dict: state_dict of the params.
    :return: list of model parameter values.
    """

    params = []
    for param in state_dict.values():
        params += list(param.cpu().detach().numpy().flatten())
    return np.array(params)


def compress_gradients(model, compression_rate: float = 1e-5, tolerance: float = 1e-2) -> np.ndarray:

    """
    Given the original gradients from one client, compress them and return its compressed gradients.

    :param model: Pytorch model of the client.
    :param compression_rate: compression rate of the compressed gradients.
    :param tolerance: tolerance parameter for computing the centers.

    :return: compressed gradients.
    """

    gradients = model.state_dict()

    linearized_gradients = get_param_values(gradients)
    output_dimension = int(len(linearized_gradients) * compression_rate)
    centers = np.random.choice(linearized_gradients, output_dimension, replace=False)
    norm_diff = np.inf

    while norm_diff > tolerance:
        distances = np.abs(linearized_gradients[:, None] - centers)
        closest_clusters = np.argmin(distances, axis=1)
        new_centers = np.array([np.mean(linearized_gradients[closest_clusters == cluster_idx])
                                for cluster_idx in range(output_dimension)])
        diff_centers = np.abs(new_centers - centers)
        norm_diff = np.linalg.norm(diff_centers)
        #print(f"Current norm diff: {norm_diff:e}", end="\r")
        centers = new_centers
    #print()

    return centers

def train_generator(generator, generator_optimizer, generator_lr_scheduler, local_model_lst, conf):
    """
    AFed Generator training
    https://arxiv.org/pdf/2501.02732
    """
    device = get_device(conf)
    generator.train()
    for i in range(50):
        generator_optimizer.zero_grad()
        a = np.random.choice([0, 1], conf["client_opt"]["batch_size"])
        a = torch.tensor(a, dtype=torch.int64).to(device)
        z_attr, eps = gen_z_attr(a, z_dim=conf["client_opt"]["AFed_generator_noise_dim"] + 2)
        fake_data = generator(z_attr.to(device))
        diversity_loss = generator.diversity_loss(eps.to(device), fake_data.to(device))
        
        # calculate the label prediction loss of the generator
        teacher_loss = 0
        client_model = model_utils.init_model(
            conf=conf,
        )
        for client_weights in local_model_lst: #!TODO: numpy to model load
            model_utils.set_weights(client_model, client_weights)
            client_model['a_cls'].eval()
            # calculate the attribute prediction loss of the generator
            a_scores = client_model['a_cls'](fake_data)
            attr_loss_g = get_loss(conf)(a_scores, a.type(torch.float32))
            teacher_loss += attr_loss_g

        generator_loss = 1 * teacher_loss + 5 * diversity_loss
        generator_loss.backward()
        generator_optimizer.step()
    generator_lr_scheduler.step()

def AFed_generator_init(conf):
    """Initialize AFed Generator model, optimizer and learning rate scheduler"""
    afed_generator = AFedGenerator(conf)
    model_path = os.path.join(
        "checkpoints",
        conf["exp_id"],
        "afed_generator"
    )
    model_utils.save_model(afed_generator, model_path)
    generator_optimizer = torch.optim.Adam(
        params=afed_generator.parameters(),
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-2,
        amsgrad=False
    )
    generator_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=generator_optimizer, gamma=0.98)
    return afed_generator, generator_optimizer, generator_lr_scheduler