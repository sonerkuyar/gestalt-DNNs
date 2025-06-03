# experiment_runner.py
import argparse
import pickle
import os
import gc
import matplotlib.pyplot as plt
import numpy as np

from ..utils.Config import Config
from ..experiment_2.distance_helper import generate_dataset_rnd
from ..utils.create_stimuli.drawing_utils import DrawShape
from ..utils.misc import config_to_path_special

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_path', type=str, required=True)
    args = parser.parse_args()

    # Load arguments from pickle
    with open(args.args_path, 'rb') as f:
        params = pickle.load(f)

    img_size = np.array((224, 224), dtype=int)
    draw_obj = DrawShape(
        background='black' if params["background"] in ['black', 'random'] else params["background"],
        img_size=img_size,
        width=10
    )

    config = Config(
        project_name='Pomerantz',
        distance_type=params["distance_type"],
        verbose=False,
        network_name=params["network_name"],
        pretraining=params["pretraining"],
        weblogger=0,
        is_pycharm='PYCHARM_HOSTED' in os.environ,
        type_ds=params["type_ds"],
        background=params["background"],
        draw_obj=draw_obj,
        rep=500,
        transf_code=params["transf_code"],
        type_ds_args=params.get("type_ds_args", None)
    )

    if params["network_name"] == 'prednet':
        config.pretraining = './models/prednet-L_0-mul-peepFalse-tbiasFalse-best.pt'

    exp_folder = f'./results/{config_to_path_special(config)}'
    generate_dataset_rnd(config, out_path=exp_folder)

    del config
    plt.close('all')
    gc.collect()

if __name__ == '__main__':
    main()
