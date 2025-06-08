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
from src.utils.misc import MyGrabNet, conver_tensor_to_plot, save_fig_pair
from copy import deepcopy
from src.utils.create_stimuli.drawing_utils import *
import torchvision


class ComputeDistance(RecordActivations):
    def get_images_for_each_category(self, dataset, N, **kwargs):
        selected_class = dataset.samples
        correct_paths = selected_class
        correct_paths = [correct_paths[i] for i in np.random.choice(range(len(correct_paths)), np.min([N, len(correct_paths)]), replace=False)]
        return correct_paths

    def compute_cosine_set(self, set, transform, path_save_fig, stats, distance_type, batch_size=32):
        distance = {}

        images = [[transform(i[0]), transform(i[1])] for i in set]
        image_plt = [[conver_tensor_to_plot(i, stats['mean'], stats['std']) for i in j] for j in images]
        if path_save_fig is not None:
            save_fig_pair(path_save_fig, image_plt, n=4)

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
                name: feat.detach().view(feat.size(0), -1)
                for name, feat in self.activation.items()
                if any(k in name for k in self.only_save)
            }

            # Forward chunk1
            _ = self.net(im1_chunk)
            second_chunk_acts = {
                name: feat.detach().view(feat.size(0), -1)
                for name, feat in self.activation.items()
                if any(k in name for k in self.only_save)
            }

            for name in first_chunk_acts:
                A, B = first_chunk_acts[name], second_chunk_acts[name]
                if distance_type == 'cossim':
                    dist_vals = torch.nn.functional.cosine_similarity(A, B, dim=1).cpu().tolist()
                else:
                    dist_vals = torch.norm(A - B, dim=1).cpu().tolist()
                distance.setdefault(name, []).extend(dist_vals)

        return distance


    def compute_random_set(self, transform, N=5, path_save_fig=None, stats=None, draw_obj=None, distance_type=None, batch_size=32):
        cossim_all = {}
        im_types = {}
        for i in range(N):
            [im_types.setdefault(k, []).append(v) for k, v in draw_obj.get_all_sets()[0].items()]

        for idx, (type, im_samples) in enumerate(im_types.items()):
            cossim_all[type] = self.compute_cosine_set(im_samples, transform, path_save_fig + f'/{type}.png', stats, distance_type, batch_size=batch_size)

        return cossim_all


def compute_distance_set(config, out_path, batch_size=32):
    config.model, norm_stats, resize = MyGrabNet().get_net(config.network_name,
                                       imagenet_pt=True if config.pretraining == 'ImageNet' else False)
    prepare_network(config.model, config, train=False)

    transf_list = [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(norm_stats['mean'], norm_stats['std'])]

    if resize:
        transf_list.insert(0, transforms.Resize(resize))

    transf_list.insert(0, transforms.Resize(299)) if config.network_name == 'inception_v3' else None
    transform = torchvision.transforms.Compose(transf_list)

    pathlib.Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    recorder = ComputeDistance(net=config.model, use_cuda=False, only_save=['Conv2d', 'Linear'])
    cossim = recorder.compute_random_set(transform=transform, N=config.rep, path_save_fig=os.path.dirname(out_path), stats=norm_stats, draw_obj=config.draw_obj, distance_type=config.distance_type, batch_size=batch_size)


    print(fg.red + f'Saved in {out_path}' + rs.fg)
    pickle.dump(cossim, open(out_path, 'wb'))


##

