"""
Random Base and Random Composite, but NOT hierarchical.
The computation is done on a List of Touple, e.g. base1 and base2, and then another List: composite1 and composite2.
The transformation across these two list match exactly, as it's pre-computed and then orderly applied.
Overall, this is used for checking proper comparison between base and composite "special" stimuli (e.g. not dots).
"""
from torchvision.transforms.functional import InterpolationMode
from sty import fg, bg, rs, ef
import pickle
import torch
from src.utils.distance_activation import RecordActivations
import os
import pathlib
from tqdm import tqdm
import torchvision.transforms as transforms
from src.utils.misc import make_cuda
from src.utils.net_utils import prepare_network, load_pretraining, GrabNet
from src.utils.misc import MyGrabNet, conver_tensor_to_plot, save_fig_pair, RandomBackground
from copy import deepcopy
from src.utils.create_stimuli.drawing_utils import *
import torchvision
from torchvision.transforms import RandomAffine, functional as F
from typing import List, Tuple
from src.utils.misc import imshow_batch

class RandomPixels(torch.nn.Module):
    def __init__(self, background_color=(0, 0, 0), line_color=(255, 255, 255)):
        super().__init__()
        self.background_color = background_color
        self.line_color = line_color

    def forward(self, input):
        i = np.array(input)
        i = i.astype(np.int16)
        s_line = len(i[i == self.line_color])
        i[i == self.line_color] = np.repeat([1000, 1000, 1000], s_line/3, axis=0).flatten()

        s = len(i[i == self.background_color])
        i[i == self.background_color] = np.random.randint(0, 255, s)

        s_line = len(i[i == [1000, 1000, 1000]])
        i[i == [1000, 1000, 1000]] = np.repeat([0, 0, 0], s_line / 3, axis=0).flatten()
        i = i.astype(np.uint8)
        plt.imshow(i)

        return transforms.ToPILImage()(i)

class ComputeDistance(RecordActivations):
    def get_images_for_each_category(self, dataset, N, **kwargs):
        selected_class = dataset.samples
        correct_paths = selected_class
        correct_paths = [correct_paths[i] for i in np.random.choice(range(len(correct_paths)), np.min([N, len(correct_paths)]), replace=False)]
        return correct_paths

    def compute_distance_set(self, set: List[Tuple[Image.Image, Image.Image]], fill_bk, transform, affine_values, path_save_fig, stats, distance_type, batch_size=32):
        distance = {}

        images = [[F.affine(i[0], *af, interpolation=InterpolationMode.NEAREST, fill=fill_bk),
                   F.affine(i[1], *af, interpolation=InterpolationMode.NEAREST, fill=fill_bk)] for af, i in zip(affine_values, set)]

        images = [[transform(i[0]), transform(i[1])] for i in images]
        image_plt = [[conver_tensor_to_plot(i, stats['mean'], stats['std']) for i in j] for j in images]
        if path_save_fig is not None:
            save_fig_pair(path_save_fig, image_plt, n=np.min([len(images), 4]))

        # Batch size for chunked GPU processing
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        im0_full = torch.stack([pair[0] for pair in images], dim=0)
        im1_full = torch.stack([pair[1] for pair in images], dim=0)
        N = im0_full.size(0)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            im0_chunk = im0_full[start:end].to(device)
            im1_chunk = im1_full[start:end].to(device)

            # Forward chunk0
            _ = self.net(im0_chunk)
            first_chunk_acts = {
                name: feat.detach().reshape(feat.size(0), -1)
                for name, feat in self.activation.items()
                if any(k in name for k in self.only_save)
            }

            # Forward chunk1
            _ = self.net(im1_chunk)
            second_chunk_acts = {
                name: feat.detach().reshape(feat.size(0), -1)
                for name, feat in self.activation.items()
                if any(k in name for k in self.only_save)
            }

            # Accumulate distances per layer
            for name in first_chunk_acts:
                A, B = first_chunk_acts[name], second_chunk_acts[name]
                if distance_type == 'cossim':
                    dist_vals = torch.nn.functional.cosine_similarity(A, B, dim=1).cpu().tolist()
                else:
                    dist_vals = torch.norm(A - B, dim=1).cpu().tolist()
                distance.setdefault(name, []).extend(dist_vals)

        return distance


    def compute_random_set(self, transform, fill_bk=None, var_tr='', N=5, type_ds=None, path_save_fig=None, stats=None, draw_obj=None, type_ds_args=None, distance_type=None):
        img_size = np.array((224, 224), dtype=int)

        def get_new_affine_values():
            tr = [np.random.uniform(-0.2, 0.2) * img_size[0], np.random.uniform(-0.2, 0.2) * img_size[1]] if 't' in var_tr else (0, 0)
            scale = np.random.uniform(0.7, 1.3) if 's' in var_tr else 1.0
            rot = np.random.uniform(0, 360) if 'r' in var_tr else 0
            return (rot, tr, scale, 0.0)

        distance_all = {}
        im_set = []
        N = 1 if var_tr == 'none' else N
        if isinstance(type_ds, str):
            tt = "".join(['_' if i == '-' else i for i in type_ds])
            for i in range(N):
                if type_ds_args is None:
                    im_set.append(getattr(draw_obj, f'get_{tt}')())
                else:
                    im_set.append(getattr(draw_obj, f'get_{tt}')(type_ds_args))
            # path_fig = path_save_fig + f'/{type_ds}.png'
            type_ds_str = type_ds
        else:  # assume type_ds is a function
            for i in range(N):
                im_set.append(type_ds[1](**type_ds_args))
            type_ds_str = type_ds[0]

        affine_values = [get_new_affine_values() for i in range(N)]

        distance_all[type_ds_str] = self.compute_distance_set(im_set, fill_bk, transform, affine_values, path_save_fig, stats, distance_type)

        return distance_all


def generate_dataset_rnd(config, out_path):
    """
    Dealing with random pixel is annoying because when applying the affine transformation we have trouble with the fill. We assume that, if we want a rndpixel background, the image is creates white-on-black. We then apply the rotation/scale etc. and only after that we make the background random pixel.
    """
    config.model, norm_stats, resize = MyGrabNet().get_net(config.network_name,
                                     imagenet_pt=True if config.pretraining == 'ImageNet' else False)
    prepare_network(config.model, config, train=False)


    transf_list = [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(norm_stats['mean'], norm_stats['std'])]

    if resize:
        transf_list.insert(0, transforms.Resize(resize))

    transform = torchvision.transforms.Compose(transf_list)
    if config.background == 'random':
        transform.transforms.insert(0, RandomPixels())

    if config.background == 'black' or config.background == 'random':
        fill_bk = 0
    elif config.background == 'white':
        fill_bk = 1
    else:
        fill_bk = config.background
    pathlib.Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

    recorder = ComputeDistance(net=config.model, use_cuda=False, only_save=['Conv2d', 'Linear'])

    distance = recorder.compute_random_set(transform=transform, fill_bk=fill_bk, var_tr=config.transf_code, N=config.rep, type_ds=config.type_ds, path_save_fig=out_path + '.png', stats=norm_stats, draw_obj=config.draw_obj, type_ds_args=config.type_ds_args, distance_type=config.distance_type)


    print(fg.red + f'Saved in {out_path}' + rs.fg)
    pickle.dump(distance, open(out_path + f'_{config.distance_type}.df', 'wb'))
    del config.model


def generate_stimuli_pairs(draw_obj, type_ds, N=5, var_tr='', type_ds_args=None):
    """
    Generate and return a list of (image_A, image_B, affine_params) tuples.
    No model or DNN evaluation is performed.
    """
    img_size = np.array((224, 224), dtype=int)

    def get_new_affine_values():
        tr = [np.random.uniform(-0.2, 0.2) * img_size[0], np.random.uniform(-0.2, 0.2) * img_size[1]] if 't' in var_tr else (0, 0)
        scale = np.random.uniform(0.7, 1.3) if 's' in var_tr else 1.0
        rot = np.random.uniform(0, 360) if 'r' in var_tr else 0
        return (rot, tr, scale, 0.0)

    im_set = []
    N = 1 if var_tr == 'none' else N
    if isinstance(type_ds, str):
        tt = "".join(['_' if i == '-' else i for i in type_ds])
        for i in range(N):
            if type_ds_args is None:
                im_pair = getattr(draw_obj, f'get_{tt}')()
            else:
                im_pair = getattr(draw_obj, f'get_{tt}')(type_ds_args)
            affine_params = get_new_affine_values()
            im_set.append((im_pair[0], im_pair[1], affine_params))
    else:  # assume type_ds is a function
        for i in range(N):
            im_pair = type_ds[1](**type_ds_args)
            affine_params = get_new_affine_values()
            im_set.append((im_pair[0], im_pair[1], affine_params))

    return im_set