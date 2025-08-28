import json
import os

import gc_utils
import h5py
import numpy as np


def create_hdf5(sim: str, it_lst: list[int], sim_dir: str):
    save_file = sim_dir + sim + "/" + sim + "_processed.hdf5"  # save location

    if not os.path.exists(save_file):
        h5py.File(save_file, "w")

    with h5py.File(save_file, "a") as hdf:
        for it in it_lst:
            it_id = gc_utils.iteration_name(it)
            data_file = sim_dir + sim + "/gc_results/interim/" + it_id + ".json"
            with open(data_file, "r") as sim_json:
                int_data = json.load(sim_json)
            data_dict = int_data[it_id]["source"]

            if it_id in hdf.keys():
                grouping = hdf[it_id]
            else:
                grouping = hdf.create_group(it_id)
            if "source" in grouping.keys():
                source = grouping["source"]
            else:
                source = grouping.create_group("source")
            for key in data_dict.keys():
                if key in source.keys():
                    del source[key]
                if key == "ptype":
                    source.create_dataset(key, data=data_dict[key])
                else:
                    source.create_dataset(key, data=np.array(data_dict[key]))
