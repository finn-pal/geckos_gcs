import argparse
import json
import os
import time

import astropy.units as u
import gc_utils
import h5py
import numpy as np
import pandas as pd
import utilities as ut
from regions import PixCoord, RectanglePixelRegion
from scipy.interpolate import griddata


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


# Conversion Functions ###############################################################################


def arcsec_deg(arcsec):
    return arcsec / 3600


def au_to_pc(au):
    return au / 206265


def abs_to_app(m_abs, d_pc):
    m_app = m_abs + 5 * np.log10(d_pc / 10)
    return np.round(m_app, 2)


def app_to_abs(m_app, d_pc):
    m_abs = m_app - 5 * np.log10(d_pc / 10)
    return np.round(m_abs, 2)


def rotate(positions, phi, inc):
    inc = np.deg2rad(inc)
    inc = np.deg2rad(90) - inc  # correct so that 90deg is edge on
    phi = np.deg2rad(phi)

    cy = np.cos(inc)
    sy = np.sin(inc)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])

    cz = np.cos(phi)
    sz = np.sin(phi)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    R = Ry @ Rz
    positions_transform = (R @ positions.T).T

    return positions_transform


def feh_to_meh(feh, alpha):
    # https://academic.oup.com/mnras/article/506/1/150/6270919?utm_source=chatgpt.com&login=true
    fa = 10**alpha
    meh = feh + np.log10(fa * 0.694 + 0.306)
    return meh


# GC Spatial Masking #################################################################################


def get_field(gal, gal_dir, view_mode="wide"):
    # muse details
    muse_field = {"narrow": 7.5, "wide": 60}  # arcsec

    # different posible regions of observation
    regions = ["central", "disk", "outflow", "ext", "far", "giant", "huge"]

    # get galaxy information and pointing details
    df_gal = pd.read_csv(gal_dir + "galaxy_details.csv")
    df_poi = pd.read_csv(gal_dir + "pointings.csv")
    df_poi.columns = df_poi.columns.str.replace(r"\s+", "", regex=True)
    df_poi["object"] = df_poi["object"].str.replace(" ", "", regex=False)

    # get galaxy details
    gal_dis_mpc = df_gal.loc[df_gal["object"] == gal, "d_mpc"].values[0]
    gal_dis_kpc = gal_dis_mpc * 1000

    # get position angle
    poi_pa = df_poi.loc[df_poi["object"] == gal, "pa_diamond"].values[0]

    field_dict = {}
    field_dict["gal_dis_kpc"] = gal_dis_kpc
    field_dict["regions"] = {}
    for region in regions:
        region_cnt = df_poi.loc[df_poi["object"] == gal, "nob_" + region].values[0]
        if region_cnt == 0:
            continue

        # get x,y offsets
        xoff_arcsec = df_poi.loc[df_poi["object"] == gal, "xoff_" + region].values[0]
        yoff_arcsec = df_poi.loc[df_poi["object"] == gal, "yoff_" + region].values[0]
        xoff_deg = np.deg2rad(arcsec_deg(xoff_arcsec))
        yoff_deg = np.deg2rad(arcsec_deg(yoff_arcsec))

        # get pointing sizings
        x_point = gal_dis_kpc * xoff_deg
        y_point = gal_dis_kpc * yoff_deg

        view_size = gal_dis_kpc * np.deg2rad(arcsec_deg(muse_field[view_mode]))

        x_l = x_point - view_size / 2
        x_u = x_point + view_size / 2

        y_l = y_point - view_size / 2
        y_u = y_point + view_size / 2

        # create field of view
        centre = np.array([x_point, y_point])
        corners_norot = np.array([[x_l, y_l], [x_u, y_l], [x_u, y_u], [x_l, y_u], [x_l, y_l]])
        corners_shifted = corners_norot - centre

        theta = np.deg2rad(poi_pa)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        corners_rot = corners_shifted @ R.T
        corners_rot += centre

        field_dict["regions"][region] = {
            "centre": centre,
            "corners_rot": corners_rot,
            "pa": poi_pa,
            "box_length": view_size,
        }

    return field_dict


def get_spatial_mask(gc_pos_rot, field_dict):
    gc_pos_msk = np.zeros(len(gc_pos_rot), dtype=bool)
    for region in field_dict["regions"]:
        centre = field_dict["regions"][region]["centre"]
        box_length = field_dict["regions"][region]["box_length"]
        angle = field_dict["regions"][region]["pa"]

        box_centre = PixCoord(*centre)
        rect = RectanglePixelRegion(
            center=box_centre, width=box_length, height=box_length, angle=angle * u.deg
        )

        positions = PixCoord(x=gc_pos_rot[:, 1], y=gc_pos_rot[:, 2])
        mask = rect.contains(positions)  # boolean array
        gc_pos_msk |= mask

    return gc_pos_msk


# GC Magnitude Masking ###############################################################################


def gc_m_abs(gc_dict, dat_dir):
    # https://www.sciencedirect.com/science/article/pii/S0019103516301014

    solar_path = dat_dir + "solar_lum.json"
    with open(solar_path, "r") as f:
        solar_dict = json.load(f)

    solar_dist_au = 1
    solar_dist_pc = au_to_pc(solar_dist_au)

    for band in gc_dict["bands"]:
        lum_var = "lum_" + band
        m_abs_var = "m_abs_pre_ex_" + band

        m_app_sol = solar_dict[band]
        m_abs_sol = app_to_abs(m_app_sol, solar_dist_pc)

        m_abs = m_abs_sol - 2.5 * np.log10(gc_dict[lum_var])

        gc_dict[m_abs_var] = m_abs

    return gc_dict


def get_luminosity(gc_dict: dict, dat_dir: str, bands: list[str] = ["I", "R"]):
    band_lst = ["U", "B", "V", "R", "I", "J", "H", "K"]

    for band in bands:
        if band not in band_lst:
            raise Warning("Band not found")

    kroupa_mlr_uni = dat_dir + "kroupa_mlr_uni.txt"
    kroupa_col = dat_dir + "kroup_columns.json"
    with open(kroupa_col, "r") as f:
        col_dict = json.load(f)

    kroupa_mlr = kroupa_mlr_uni
    df = pd.read_csv(kroupa_mlr, sep=r"\s+", header=None, comment="#", names=col_dict)
    M_H_k = df["M_H"].values
    age_k = df["Age"].values

    age = gc_dict["age"]
    feh = gc_dict["feh"]

    # can use assumptions from
    # https://academic.oup.com/mnras/article/507/4/5882/6328491
    # or just assume alpha enhancement = 0
    alphafe = 0
    meh = feh_to_meh(feh, alphafe)

    for band in bands:
        ML_var = "ML_" + band
        ML_band = df[ML_var].values

        ML_cubic = griddata(points=(M_H_k, age_k), values=ML_band, xi=(meh, age), method="cubic")
        ML_near = griddata(points=(M_H_k, age_k), values=ML_band, xi=(meh, age), method="nearest")

        # Find NaNs (outside convex hull) and fill using nearest-neighbor
        nan_mask = np.isnan(ML_cubic)
        ML_com = ML_cubic
        ML_com[nan_mask] = ML_near[nan_mask]

        lum_var = "lum_" + band
        lum_band = gc_dict["mass"] / ML_com

        gc_dict[ML_var] = ML_com
        gc_dict[lum_var] = lum_band

    gc_dict["bands"] = bands
    gc_dict = gc_m_abs(gc_dict, dat_dir)

    return gc_dict


def los_mask(observer, poi, gas_pos_rot, gas_l):
    OP = poi - observer
    OP_len2 = np.dot(OP, OP)
    OP_len = np.sqrt(OP_len2)

    OG = gas_pos_rot - observer  # shape (N, 3)

    # Projection of OG onto OP (parameter t along the line)
    t = (OG @ OP) / OP_len2

    # Squared perpendicular distance using vector identity:
    # |a x b|^2 = |a|^2|b|^2 - (a·b)^2
    OP_dot_OG = OG @ OP
    OG_len2 = np.einsum("ij,ij->i", OG, OG)  # row-wise squared norms

    # cross_sq = np.maximum(cross_sq, 0.0)
    cross_sq = OG_len2 * OP_len2 - OP_dot_OG**2

    # cross_sq >= 0 but rounding errors can cause slightly negative values
    # the following ensures positive values
    cross_sq = np.maximum(cross_sq, 0.0)

    dist_perp = np.sqrt(cross_sq) / OP_len

    g_msk = (t >= 0.0) & (t <= 1.0) & (dist_perp < gas_l)
    return g_msk


def dust(part, observer, poi, phi, inc, box_correction=2 / 3, q_dust=2.5e22):
    gas_pos = part["gas"].prop("host.distance.principle")
    gas_pos_rot = rotate(gas_pos, phi, inc)
    gas_l = part["gas"]["size"]

    g_msk = los_mask(observer, poi, gas_pos_rot, gas_l)

    # Slice once
    gas_l_sel = gas_l[g_msk]
    n_H_total = part["gas"].prop("number.density")[g_msk]
    ntrl_frac = part["gas"].prop("hydrogen.neutral.fraction")[g_msk]
    z_metal_local = part["gas"].prop("massfraction")[g_msk, 0]  # better indexing

    # Precompute constants
    z_metal_solar = ut.constant.sun_protosolar_metals_mass_fraction
    cm_per_kpc = ut.constant.cm_per_kpc

    # Combine ops to avoid temporaries
    n_H_neutral = n_H_total * ntrl_frac
    leng_cm = gas_l_sel * cm_per_kpc * box_correction
    z_rat = z_metal_local / z_metal_solar

    # Column density
    cd = np.dot(n_H_neutral * z_rat, leng_cm)  # dot is faster than sum of product
    E_BV = cd / q_dust
    return E_BV


def k_prime(wavelength: float, R_Vprime: float = 4.05):
    # https://ui.adsabs.harvard.edu/abs/2000ApJ...533..682C/abstract
    # wavelength in micro metres

    if (0.63 <= wavelength) & (wavelength <= 2.20):
        k_p = 2.659 * (-1.857 + 1.040 / wavelength) + R_Vprime
    elif (0.12 <= wavelength) & (wavelength < 0.63):
        k_p = 2.659 * (-2.156 + 1.509 / wavelength - 0.198 / wavelength**2 + 0.011 / wavelength**3) + R_Vprime
    else:
        raise RuntimeError("Wavelength outside extinction region")

    return k_p


def get_extinction(E_BV: float, dat_dir: str, band: str):
    # https://ui.adsabs.harvard.edu/abs/2000ApJ...533..682C/abstract
    # https://www.oxfordreference.com/display/10.1093/acref/9780199609055.001.0001/acref-9780199609055-e-2038
    # https://www.sciencedirect.com/science/article/pii/S0019103516301014

    bands_path = dat_dir + "bands.json"
    with open(bands_path, "r") as f:
        bands_dict = json.load(f)

    wavelength = bands_dict[band]["effective_wavelength"]

    E_BVs = 0.44 * E_BV
    k_p = k_prime(wavelength)
    A = k_p * E_BVs

    return A


def get_magnitudes(part, gc_dict, field_dict, phi, inc, dat_dir: str, bands: list[str] = ["I", "R"]):
    observer = np.array([field_dict["gal_dis_kpc"], 0, 0])
    # dist_pc = field_dict["gal_dis_kpc"] * 1000

    gc_pos_rot = gc_dict["pos_rot"]
    op_dist_kpc = np.linalg.norm(observer - gc_pos_rot)
    op_dist_pc = op_dist_kpc * 1000

    gc_dict = get_luminosity(gc_dict, dat_dir, bands)

    for band in gc_dict["bands"]:
        m_abs_var = "m_abs_pre_ex_" + band
        m_app_var = "m_app_pre_ex_" + band

        gc_dict[m_app_var] = abs_to_app(gc_dict[m_abs_var], op_dist_pc)

        # get extinctions
        ex_var = "A_" + band

        m_abs_ex_var = "m_abs_ex_" + band
        m_app_ex_var = "m_app_ex_" + band

        A_lst = []
        for poi in gc_pos_rot:
            EB_V = dust(part, observer, poi, phi, inc)
            A = get_extinction(EB_V, dat_dir, band)
            A_lst.append(A)
        A_lst = np.array(A_lst)

        gc_dict[ex_var] = A_lst
        gc_dict[m_app_ex_var] = gc_dict[m_app_var] + A_lst
        gc_dict[m_abs_ex_var] = app_to_abs(gc_dict[m_app_ex_var], op_dist_pc)

    return gc_dict


def get_mag_mask(gc_dict, mag_lim, band="I", extinction: bool = True):
    if extinction:
        m_var = "m_abs_ex_" + band
    else:
        m_var = "m_abs_pre_ex_" + band
    mag_msk = gc_dict[m_var] < mag_lim
    gc_dict["mag_msk"] = mag_msk

    return gc_dict


# GC Properties ######################################################################################


def get_gc_properties(sim, sim_dir, it, snap):
    all_data = sim_dir + sim + "/" + sim + "_res7100/snapshot_times.txt"
    all_snaps = pd.read_table(all_data, comment="#", header=None, sep=r"\s+")
    all_snaps.columns = [
        "index",
        "scale_factor",
        "redshift",
        "time_Gyr",
        "lookback_time_Gyr",
        "time_width_Myr",
    ]
    snp_tim = all_snaps[all_snaps["index"] == snap]["time_Gyr"].values[0]

    proc_file = sim_dir + sim + "/" + sim + "_processed.hdf5"
    proc_data = h5py.File(proc_file, "r")  # open processed data file

    it_id = gc_utils.iteration_name(it)
    snap_id = gc_utils.snapshot_name(snap)

    snap_dat = proc_data[it_id]["snapshots"][snap_id]
    gcids_snp = snap_dat["gc_id"][()]
    mass_snp = 10 ** snap_dat["mass"][()]
    pos_xyz_snp = snap_dat["pos.xyz"][()]
    # circ_snp = snap_dat["lz_norm"][()] # remove for katana
    group_snp = snap_dat["group_id"][()]
    acc_snp = np.array([0 if group == 0 else 1 for group in group_snp])

    src_dat = proc_data[it_id]["source"]
    ana_msk = src_dat["analyse_flag"][()] == 1
    gcids_src = src_dat["gc_id"][ana_msk]
    feh_src = src_dat["feh"][ana_msk]
    form_src = src_dat["form_time"][ana_msk]

    feh_lst = []
    age_lst = []
    for gcid in gcids_snp:
        gcidx = np.where(gcids_src == gcid)[0][0]
        feh = feh_src[gcidx]
        age = snp_tim - form_src[gcidx]

        feh_lst.append(feh)
        age_lst.append(age)

    gc_dict = {
        "gcid": gcids_snp,
        "mass": mass_snp,
        "feh": np.array(feh_lst),
        "age": np.array(age_lst),
        # "circ": circ_snp,
        "pos": pos_xyz_snp,
        "acc": acc_snp,
    }

    return gc_dict


def process_data(
    sim: str,
    sim_dir: str,
    gal: str,
    gal_dir: str,
    dat_dir: str,
    it: int,
    snap: int,
    phi: float,
    inc: float,
    bands: list[str] = ["I", "R"],
    mag_lim_band: str = "I",
    mag_lim: float = -6,
):
    fir_dir = sim_dir + sim + "/" + sim + "_res7100"
    part = gc_utils.open_snapshot(snap, fir_dir, ["star", "gas"])

    gc_dict = get_gc_properties(sim, sim_dir, it, snap)
    gc_dict["pos_rot"] = rotate(gc_dict["pos"], phi, inc)

    field_dict = get_field(gal, gal_dir)
    gc_dict["pos_msk"] = get_spatial_mask(gc_dict["pos_rot"], field_dict)

    gc_dict = get_magnitudes(part, gc_dict, field_dict, phi, inc, dat_dir, bands)
    gc_dict = get_mag_mask(gc_dict, mag_lim, band=mag_lim_band, extinction=True)

    return gc_dict


# Convert Dictionary to Saveable Format ##############################################################


def convert_ndarrays(obj):
    # Convert all numpy arrays to lists recursively
    if isinstance(obj, dict):
        return {k: convert_ndarrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# Main Sequence ######################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", required=True, type=str, help="simulation name (e.g. m12i)")
    parser.add_argument("-g", "--galaxy", required=True, type=str, help="gexkos galaxy name (e.g. NGC3957)")
    parser.add_argument("-l", "--location", required=False, type=str, help="data location", default="local")

    parser.add_argument("-a", "--iteration", required=False, type=int, help="sim iteration", default=0)
    parser.add_argument("-b", "--snapshot", required=False, type=int, help="sim snapshot", default=600)

    parser.add_argument("-p", "--phi", required=False, type=int, help="azimuthal angle", default=0)
    parser.add_argument("-i", "--inclination", required=False, type=int, help="inclination", default=90)

    args = parser.parse_args()

    sim = args.simulation
    loc = args.location
    it = args.iteration
    snap = args.snapshot

    phi = args.phi
    inc = args.inclination

    gal = args.galaxy

    sim_dir = get_dir(loc, "simulation")
    gal_dir = get_dir(loc, "galaxy")
    dat_dir = get_dir(loc, "data")

    print("Retrieving " + gal + " on " + sim)
    print("it: " + str(it) + ", snap: " + str(snap) + ", phi: " + str(phi) + ", i: " + str(inc))

    start_time = time.time()

    gc_dict = process_data(sim, sim_dir, gal, gal_dir, dat_dir, it, snap, phi, inc)
    gc_dict_serializable = convert_ndarrays(gc_dict)

    end_time = time.time()
    print("time:", np.round((end_time - start_time) / 60, 2), "min")

    save_dir = gal_dir + gal + "/" + sim  # save location
    it_dir = save_dir + "/" + "iterations"
    if not os.path.exists(it_dir):
        os.makedirs(it_dir)

    # Save dictionary to JSON file
    it_id = gc_utils.iteration_name(it)
    it_file = it_dir + "/" + it_id + "_p" + str(phi) + "_i" + str(inc) + ".json"
    with open(it_file, "w") as f:
        json.dump(gc_dict_serializable, f, indent=4)  # indent=4 makes it human-readable
