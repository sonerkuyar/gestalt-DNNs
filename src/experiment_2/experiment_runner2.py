# experiment_runner2.py
import argparse
import pickle
import os
import gc
import matplotlib.pyplot as plt
import numpy as np

from src.utils.Config import Config
from src.experiment_2.distance_helper import ComputeDistance
from src.utils.create_stimuli.drawing_utils import DrawShape
from src.utils.misc import config_to_path_special

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stimuli_path', type=str, required=True, help='Path to stimuli .pkl file')
    parser.add_argument('--network_name', type=str, required=True)
    parser.add_argument('--pretraining', type=str, required=True)
    parser.add_argument('--background', type=str, required=True)
    parser.add_argument('--distance_type', type=str, required=True)
    parser.add_argument('--transf_code', type=str, required=True)
    parser.add_argument('--type_ds', type=str, required=True)
    parser.add_argument('--type_ds_args', type=str, default=None)
    args = parser.parse_args()

    # Load stimuli
    with open(args.stimuli_path, 'rb') as f:
        stimuli = pickle.load(f)

    img_size = np.array((224, 224), dtype=int)
    draw_obj = DrawShape(
        background='black' if args.background in ['black', 'random'] else args.background,
        img_size=img_size,
        width=10
    )

    config = Config(
        project_name='Pomerantz',
        distance_type=args.distance_type,
        verbose=False,
        network_name=args.network_name,
        pretraining=args.pretraining,
        weblogger=0,
        is_pycharm='PYCHARM_HOSTED' in os.environ,
        type_ds=args.type_ds,
        background=args.background,
        draw_obj=draw_obj,
        rep=len(stimuli),
        transf_code=args.transf_code,
        type_ds_args=args.type_ds_args
    )

    # Prepare model and transforms
    from src.experiment_2.distance_helper import MyGrabNet, prepare_network
    config.model, norm_stats, resize = MyGrabNet().get_net(
        config.network_name,
        imagenet_pt=True if config.pretraining == 'ImageNet' else False
    )
    prepare_network(config.model, config, train=False)

    import torchvision
    from torchvision import transforms
    transf_list = [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(norm_stats['mean'], norm_stats['std'])]
    if resize:
        transf_list.insert(0, transforms.Resize(resize))
    transform = torchvision.transforms.Compose(transf_list)
    if config.background == 'random':
        from src.experiment_2.distance_helper import RandomPixels
        transform.transforms.insert(0, RandomPixels())

    fill_bk = 0 if config.background in ['black', 'random'] else (1 if config.background == 'white' else config.background)

    recorder = ComputeDistance(net=config.model, use_cuda=False, only_save=['Conv2d', 'Linear'])

    # Compute distances for all pairs in the loaded stimuli
    im_set = [(im0, im1) for (im0, im1, _) in stimuli]
    affine_values = [aff for (_, _, aff) in stimuli]
    distance = recorder.compute_distance_set(
        im_set, fill_bk, transform, affine_values, None, norm_stats, config.distance_type
    )

    # Save results
    exp_folder = f'./results/{config_to_path_special(config)}'
    os.makedirs(os.path.dirname(exp_folder), exist_ok=True)
    pickle.dump(distance, open(exp_folder + f'_{config.distance_type}.df', 'wb'))
    print(f'Saved in {exp_folder}_{config.distance_type}.df')

    del config
    plt.close('all')
    gc.collect()

if __name__ == '__main__':
    main()