"""
Random Base and Random Composite, but NOT hierarchical.
The computation is done on a List of Tuple, e.g. base1 and base2, and then another List: composite1 and composite2.
The transformation across these two lists matches exactly, as it’s pre-computed and then orderly applied.
Overall, this is used for checking proper comparison between base and composite "special" stimuli (e.g. not dots).
"""
import os
import pickle
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import functional as F
from sty import fg, bg, rs, ef
from tqdm import tqdm

from typing import List, Tuple
from copy import deepcopy

from src.utils.distance_activation import RecordActivations
from src.utils.net_utils import prepare_network, load_pretraining, GrabNet
from src.utils.misc import (
    make_cuda,
    MyGrabNet,
    conver_tensor_to_plot,
    save_fig_pair,
    RandomBackground,
    imshow_batch,
)
from src.utils.create_stimuli.drawing_utils import *


class RandomPixels(torch.nn.Module):
    def __init__(self, background_color=(0, 0, 0), line_color=(255, 255, 255)):
        super().__init__()
        self.background_color = background_color
        self.line_color = line_color

    def forward(self, input):
        i = np.array(input).astype(np.int16)
        # mark line pixels
        mask_line = (i == self.line_color).all(axis=-1)
        s_line = mask_line.sum()
        i[mask_line] = np.repeat([1000, 1000, 1000], s_line, axis=0)
        # randomize background pixels
        mask_back = (i == self.background_color).all(axis=-1)
        s_back = mask_back.sum()
        i[mask_back] = np.random.randint(0, 255, s_back)
        # restore line pixels
        mask_temp = (i == [1000, 1000, 1000]).all(axis=-1)
        s_temp = mask_temp.sum()
        i[mask_temp] = np.repeat([0, 0, 0], s_temp, axis=0)
        i = i.astype(np.uint8)
        plt.imshow(i)
        return transforms.ToPILImage()(i)


class ComputeDistance(RecordActivations):
    def get_images_for_each_category(self, dataset, N, **kwargs):
        samples = dataset.samples
        choice = np.random.choice(len(samples), min(N, len(samples)), replace=False)
        return [samples[i] for i in choice]

    def compute_distance_set(
        self,
        set: List[Tuple[Image.Image, Image.Image]],
        fill_bk,
        transform,
        affine_values,
        path_save_fig,
        stats,
        distance_type,
    ):
        distance = {}

        # 1) affine + transform each pair
        images = [
            [
                F.affine(img0, *af, interpolation=InterpolationMode.NEAREST, fill=fill_bk),
                F.affine(img1, *af, interpolation=InterpolationMode.NEAREST, fill=fill_bk),
            ]
            for (img0, img1), af in zip(set, affine_values)
        ]
        images = [[transform(a), transform(b)] for a, b in images]

        # optional preview
        image_plt = [
            [conver_tensor_to_plot(x, stats["mean"], stats["std"]) for x in pair]
            for pair in images
        ]
        if path_save_fig:
            save_fig_pair(path_save_fig, image_plt, n=min(len(images), 4))

        # 2) batch into chunks to avoid OOM
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        im0_batch = torch.stack([p[0] for p in images], dim=0).to(device)
        im1_batch = torch.stack([p[1] for p in images], dim=0).to(device)

        batch_size = 16  # tune to fit your GPU
        Ntotal = im0_batch.size(0)
        for start in range(0, Ntotal, batch_size):
            end = min(start + batch_size, Ntotal)
            c0 = im0_batch[start:end]
            c1 = im1_batch[start:end]

            # forward first chunk
            _ = self.net(c0)
            first_acts = {
                name: feat.detach().view(feat.size(0), -1)
                for name, feat in self.activation.items()
                if any(k in name for k in self.only_save)
            }

            # forward second chunk
            _ = self.net(c1)
            second_acts = {
                name: feat.detach().view(feat.size(0), -1)
                for name, feat in self.activation.items()
                if any(k in name for k in self.only_save)
            }

            # compute and accumulate
            for name in first_acts:
                A, B = first_acts[name], second_acts[name]
                if distance_type == "cossim":
                    vals = torch.nn.functional.cosine_similarity(A, B, dim=1).cpu().tolist()
                else:
                    vals = torch.norm(A - B, dim=1).cpu().tolist()
                distance.setdefault(name, []).extend(vals)

        return distance

    def compute_random_set(
        self,
        transform,
        fill_bk=None,
        var_tr="",
        N=5,
        type_ds=None,
        path_save_fig=None,
        stats=None,
        draw_obj=None,
        type_ds_args=None,
        distance_type=None,
    ):
        img_size = np.array((224, 224), dtype=int)

        def get_new_affine_values():
            tr = (
                [np.random.uniform(-0.2, 0.2) * img_size[0], np.random.uniform(-0.2, 0.2) * img_size[1]]
                if "t" in var_tr
                else (0, 0)
            )
            scale = np.random.uniform(0.7, 1.3) if "s" in var_tr else 1.0
            rot = np.random.uniform(0, 360) if "r" in var_tr else 0
            return (rot, tr, scale, 0.0)

        distance_all = {}
        im_set = []
        N_iter = 1 if var_tr == "none" else N

        if isinstance(type_ds, str):
            key = "".join(["_" if c == "-" else c for c in type_ds])
            for _ in range(N_iter):
                im = getattr(draw_obj, f"get_{key}")() if type_ds_args is None else getattr(draw_obj, f"get_{key}")(type_ds_args)
                im_set.append(im)
            ds_key = type_ds
        else:
            for _ in range(N_iter):
                im_set.append(type_ds[1](**type_ds_args))
            ds_key = type_ds[0]

        affs = [get_new_affine_values() for _ in range(N_iter)]
        distance_all[ds_key] = self.compute_distance_set(im_set, fill_bk, transform, affs, path_save_fig, stats, distance_type)
        return distance_all


def generate_dataset_rnd(config, out_path):
    config.model, norm_stats, resize = MyGrabNet().get_net(
        config.network_name, imagenet_pt=(config.pretraining == "ImageNet")
    )
    prepare_network(config.model, config, train=False)

    transf_list = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(norm_stats["mean"], norm_stats["std"]),
    ]
    if resize:
        transf_list.insert(0, transforms.Resize(resize))
    transform = transforms.Compose(transf_list)
    if config.background == "random":
        transform.transforms.insert(0, RandomPixels())

    fill_bk = 0 if config.background in ["black", "random"] else (1 if config.background == "white" else config.background)
    pathlib.Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

    recorder = ComputeDistance(net=config.model, use_cuda=False, only_save=["Conv2d", "Linear"])
    distance = recorder.compute_random_set(
        transform=transform,
        fill_bk=fill_bk,
        var_tr=config.transf_code,
        N=config.rep,
        type_ds=config.type_ds,
        path_save_fig=out_path + ".png",
        stats=norm_stats,
        draw_obj=config.draw_obj,
        type_ds_args=config.type_ds_args,
        distance_type=config.distance_type,
    )
    print(fg.red + f"Saved in {out_path}" + rs.fg)
    pickle.dump(distance, open(out_path + f"_{config.distance_type}.df", "wb"))
    del config.model


def generate_stimuli_pairs(draw_obj, type_ds, N=5, var_tr="", type_ds_args=None):
    img_size = np.array((224, 224), dtype=int)

    def get_new_affine_values():
        tr = (
            [np.random.uniform(-0.2, 0.2) * img_size[0], np.random.uniform(-0.2, 0.2) * img_size[1]]
            if "t" in var_tr
            else (0, 0)
        )
        scale = np.random.uniform(0.7, 1.3) if "s" in var_tr else 1.0
        rot = np.random.uniform(0, 360) if "r" in var_tr else 0
        return (rot, tr, scale, 0.0)

    im_set = []
    N_iter = 1 if var_tr == "none" else N
    if isinstance(type_ds, str):
        key = "".join(["_" if c == "-" else c for c in type_ds])
        for _ in range(N_iter):
            im_pair = getattr(draw_obj, f"get_{key}")() if type_ds_args is None else getattr(draw_obj, f"get_{key}")(type_ds_args)
            aff = get_new_affine_values()
            im_set.append((im_pair[0], im_pair[1], aff))
    else:
        for _ in range(N_iter):
            im_pair = type_ds[1](**type_ds_args)
            aff = get_new_affine_values()
            im_set.append((im_pair[0], im_pair[1], aff))
    return im_set