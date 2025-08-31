import argparse
import json
import os

import gc_utils
import h5py
import numpy as np


def create_hdf5(save_file: str, it_dir: str, it_lst: list[int]):
    if not os.path.exists(save_file):
        h5py.File(save_file, "w")

    with h5py.File(save_file, "a") as hdf:
        for it in it_lst:
            it_id = gc_utils.iteration_name(it)
            it_file = it_dir + "/" + it_id + "_p" + str(phi) + "_i" + str(inc) + ".json"
            with open(it_file, "r") as sim_json:
                it_data = json.load(sim_json)

            if it_id in hdf.keys():
                grouping = hdf[it_id]
            else:
                grouping = hdf.create_group(it_id)

            for key, value in it_data.items():
                if key in grouping.keys():
                    del grouping[key]

                # Handle lists of "true"/"false" strings → bool
                if isinstance(value, list) and all(
                    isinstance(v, str) and v.lower() in ("true", "false") for v in value
                ):
                    bool_array = np.array([v.lower() == "true" for v in value], dtype=bool)
                    grouping.create_dataset(key, data=bool_array)

                # Handle list of strings properly (convert to Python str, avoid <U)
                elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                    str_dt = h5py.string_dtype(encoding="utf-8")
                    # Coerce to list of Python strings (not NumPy <U)
                    str_list = [str(v) for v in value]
                    grouping.create_dataset(key, data=str_list, dtype=str_dt)

                else:
                    arr = np.array(value)

                    # Fallback: if it's an object or string-like array, convert manually
                    if arr.dtype.kind in {"U", "O"}:
                        str_dt = h5py.string_dtype(encoding="utf-8")
                        grouping.create_dataset(key, data=arr.astype(str), dtype=str_dt)
                    else:
                        grouping.create_dataset(key, data=arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", required=True, type=str, help="simulation name (e.g. m12i)")
    parser.add_argument("-g", "--galaxy", required=True, type=str, help="gexkos galaxy name (e.g. NGC3957)")
    parser.add_argument("-l", "--location", required=False, type=str, help="data location", default="local")

    parser.add_argument("-a", "--it_low_limit", required=True, type=int, help="lower bound it", default=0)
    parser.add_argument("-b", "--it_up_limit", required=True, type=int, help="upper bound it", default=100)
    parser.add_argument("-c", "--snapshot", required=False, type=int, help="sim snapshot", default=600)

    parser.add_argument("-p", "--phi", required=False, type=int, help="azimuthal angle", default=0)
    parser.add_argument("-i", "--inclination", required=False, type=int, help="inclination", default=90)

    args = parser.parse_args()

    sim = args.simulation
    loc = args.location
    it_min = args.it_low_limit
    it_max = args.it_up_limit
    snap = args.snapshot

    phi = args.phi
    inc = args.inclination

    gal = args.galaxy

    if loc == "local":
        gal_dir = "../../geckos/"
    elif loc == "katana":
        gal_dir = "/srv/scratch/astro/z5114326/geckos/"
    elif loc == "expansion":
        gal_dir = "/Volumes/Expansion/geckos/"
    else:
        raise RuntimeError("Incorrect galaxy location provided. Must be local, katana or expansion")

    snap_id = gc_utils.snapshot_name(snap)
    it_lst = np.arange(it_min, it_max + 1, dtype=int)

    save_dir = gal_dir + gal + "/" + sim  # save location
    it_dir = save_dir + "/" + "iterations"

    save_file = save_dir + "/" + sim + "_" + gal + "_" + snap_id + "_p" + str(phi) + "_i" + str(inc) + ".hdf5"

    create_hdf5(save_file, it_dir, it_lst)
