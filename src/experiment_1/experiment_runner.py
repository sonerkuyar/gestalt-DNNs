# src/experiment_1/experiment_runner.py
import argparse
import pickle
import os
import gc
import matplotlib.pyplot as plt
import numpy as np
import random

from src.utils.Config import Config
from src.experiment_1.distance_helper import compute_distance_set
from src.utils.create_stimuli.drawing_utils import DrawShape
from src.utils.misc import config_to_path_hierarchical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_path', type=str, required=True,
                        help="Path to a pickle file containing a dict with keys "
                             "['network_name','pretraining','background','distance_type']")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    args = parser.parse_args()

    # load the arguments dict
    with open(args.args_path, 'rb') as f:
        params = pickle.load(f)

    # seed everything for reproducibility
    torch_seed = 0
    import torch
    torch.manual_seed(torch_seed)
    np.random.seed(torch_seed)
    random.seed(torch_seed)

    # build the Config
    img_size = np.array((224, 224), dtype=int)
    config = Config(
        project_name='Pomerantz',
        verbose=False,
        distance_type=params["distance_type"],
        background=params["background"],
        network_name=params["network_name"],
        pretraining=params["pretraining"],
        weblogger=0,
        is_pycharm='PYCHARM_HOSTED' in os.environ,
        rep=500,
        draw_obj=DrawShape(
            background=params["background"],
            img_size=img_size,
            width=10,
            min_dist_bw_points=20,
            min_dist_borders=40
        )
    )

    # special prednet path override if needed
    if params["network_name"] == 'prednet':
        config.pretraining = './models/prednet-L_0-mul-peepFalse-tbiasFalse-best.pt'

    # compute and save
    exp_folder = f'./results/{config_to_path_hierarchical(config)}'
    out_file = exp_folder + f'{params["distance_type"]}.df'
    compute_distance_set(config, out_path=out_file, batch_size=args.batch_size)

    # cleanup
    del config
    plt.close('all')
    gc.collect()


if __name__ == '__main__':
    main()