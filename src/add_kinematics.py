import argparse
import json
import os

import agama
import gc_utils
import h5py
import numpy as np


def get_dir(loc_: str, file_type: str):
    if file_type == "simulation":
        if loc_ == "local":
            dir_ = "../../simulations/"
        elif loc_ == "katana":
            dir_ = "/srv/scratch/astro/z5114326/simulations/"
        elif loc_ == "expansion":
            dir_ = "/Volumes/Expansion/simulations/"
        else:
            raise RuntimeError("Incorrect simulation location provided. Must be local, katana or expansion")

    elif file_type == "galaxy":
        if loc_ == "local":
            dir_ = "../../geckos/"
        elif loc_ == "katana":
            dir_ = "/srv/scratch/astro/z5114326/geckos/"
        elif loc_ == "expansion":
            dir_ = "/Volumes/Expansion/geckos/"
        else:
            raise RuntimeError("Incorrect galaxy location provided. Must be local, katana or expansion")

    elif file_type == "data":
        if loc_ == "local":
            dir_ = "data/"
        elif loc_ == "katana":
            dir_ = "/srv/scratch/astro/z5114326/geckos_gcs/data/"
        elif loc_ == "expansion":
            dir_ = "/Volumes/Expansiongeckos_gcs/data/"
        else:
            raise RuntimeError("Incorrect galaxy location provided. Must be local, katana or expansion")

    else:
        raise RuntimeError("Incorrect file_type provided. Must be sim or gal")

    return dir_


def add_data(
    proc_data,
    gec_data,
    snap: int,
    it_lst: list[int],
):
    # NEED TO UPDATE TO ACCOUNT FOR GECKOS FORMAT
    agama.setUnits(length=1, velocity=1, mass=1)
    pot_file = sim_dir + sim + "/potentials/snap_%d/combined_snap_%d.ini" % (snap, snap)
    pot = agama.Potential(pot_file)

    snap_id = gc_utils.snapshot_name(snap)
    for it in it_lst:
        it_id = gc_utils.iteration_name(it)
        snp_dat = proc_data[it_id]["snapshots"][snap_id]
        pos_xyz = snp_dat["pos.xyz"][()]
        vel_xyz = snp_dat["vel.xyz"][()]

        eccentricity = snp_dat["ecc"][()]
        circ = snp_dat["lz_norm"][()]

        ic_car = np.hstack((pos_xyz, vel_xyz))
        orbits = agama.orbit(potential=pot, ic=ic_car, time=10 * pot.Tcirc(ic_car), trajsize=10000)

        max_z = []
        for orbit in orbits:
            max_z.append(np.max(np.abs(orbit[1][:, 2])))
        max_z = np.array(max_z)

        # addition of 0.01 as sometimes rounding error lead to max_z not being max point
        check_z = np.abs(ic_car[:, 2]) > max_z + 0.01
        if len(np.where(check_z)[0]) > 0:
            raise ValueError("Issues with z_max")

        # z_flag = (np.abs(ic_car[:, 2]) > np.array(max_z)).astype(int)

        # add to hdf5
        if it_id in gec_data.keys():
            it_grouping = gec_data[it_id]
        else:
            it_grouping = gec_data.create_group(it_id)

        prop_lst = ["circ", "eccentricity", "max_z", "max_z_flag"]
        for prop in prop_lst:
            if prop in it_grouping.keys():
                del it_grouping[prop]

        it_grouping.create_dataset("circ", data=circ)
        it_grouping.create_dataset("eccentricity", data=eccentricity)
        it_grouping.create_dataset("max_z", data=max_z)
        # it_grouping.create_dataset("max_z_flag", data=z_flag)

    gec_data.close()
    proc_data.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", required=True, type=str, help="simulation name (e.g. m12i)")
    parser.add_argument("-g", "--galaxy", required=True, type=str, help="gexkos galaxy name (e.g. NGC3957)")
    parser.add_argument("-l", "--location", required=False, type=str, help="data location", default="local")

    parser.add_argument("-a", "--it_low_limit", required=False, type=int, help="lower bound it", default=0)
    parser.add_argument("-b", "--it_up_limit", required=False, type=int, help="upper bound it", default=100)
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

    sim_dir = get_dir(loc, "simulation")
    gal_dir = get_dir(loc, "galaxy")

    snap_id = gc_utils.snapshot_name(snap)
    it_lst = np.arange(it_min, it_max + 1, dtype=int)

    save_dir = gal_dir + gal + "/" + sim  # save location
    it_dir = save_dir + "/" + "iterations"

    gec_file = save_dir + "/" + sim + "_" + gal + "_" + snap_id + "_p" + str(phi) + "_i" + str(inc) + ".hdf5"
    gec_data = h5py.File(gec_file, "a")  # open processed data file

    proc_file = sim_dir + sim + "/" + sim + "_processed.hdf5"
    proc_data = h5py.File(proc_file, "r")  # open processed data file

    add_data(proc_data, gec_data, snap, it_lst)
