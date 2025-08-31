import agama
import numpy as np

agama.setUnits(length=1, velocity=1, mass=1)


# need to get velocities and circularities


def convert_to_coords(init_cond, coord="cylindrical"):
    """
    Convert Cartesian initial conditions to cylindrical or spherical coordinates.

    Parameters:
        init_cond : ndarray of shape (N, 6)
            Each row: [x, y, z, vx, vy, vz]
        coord : str
            Either 'cylindrical' or 'spherical'

    Returns:
        coords : ndarray of shape (N, 6)
            Cylindrical: [R, phi, z, v_R, v_phi, v_z]
            Spherical:   [r, theta, phi, v_r, v_theta, v_phi]
    """
    x, y, z = init_cond[:, 0], init_cond[:, 1], init_cond[:, 2]
    vx, vy, vz = init_cond[:, 3], init_cond[:, 4], init_cond[:, 5]

    if coord == "cylindrical":
        R = np.hypot(x, y)
        phi = np.arctan2(y, x)
        v_R = (x * vx + y * vy) / R
        v_phi = (-y * vx + x * vy) / R
        return np.column_stack((R, phi, z, v_R, v_phi, vz))

    elif coord == "spherical":
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)  # polar angle
        phi = np.arctan2(y, x)

        # Unit vectors
        r_hat = np.stack((x, y, z), axis=1) / r[:, None]
        theta_hat = np.stack(
            (
                x * z / (r**2 * np.sqrt(x**2 + y**2)),
                y * z / (r**2 * np.sqrt(x**2 + y**2)),
                -np.sqrt(x**2 + y**2) / r**2,
            ),
            axis=1,
        )
        phi_hat = np.stack((-y, x, np.zeros_like(z)), axis=1) / (x**2 + y**2)[:, None] ** 0.5

        vel = np.stack((vx, vy, vz), axis=1)

        v_r = np.sum(vel * r_hat, axis=1)
        v_theta = np.sum(vel * theta_hat, axis=1)
        v_phi = np.sum(vel * phi_hat, axis=1)

        return np.column_stack((r, theta, phi, v_r, v_theta, v_phi))

    else:
        raise ValueError("coord must be 'cylindrical' or 'spherical'")


def get_orbits(
    df,
    POTENTIAL_TYPE="PriceWhelan22.ini",
    SAVE_TABLE_NAME="orbit_table_full",
    DATA_FOLDER="data/",
    POTENTIAL_FOLDER="potentials/",
):
    # NEED TO UPDATE TO ACCOUNT FOR GECKOS FORMAT

    ic_car = np.vstack((df["x_gc"], df["y_gc"], df["z_gc"], df["u"], df["v"], df["w"])).T
    ic_cyl = convert_to_coords(ic_car, coord="cylindrical")
    ic_sph = convert_to_coords(ic_car, coord="spherical")

    lx = ic_car[:, 1] * ic_car[:, 5] - ic_car[:, 2] * ic_car[:, 4]
    ly = ic_car[:, 2] * ic_car[:, 3] - ic_car[:, 0] * ic_car[:, 5]
    lz = ic_car[:, 0] * ic_car[:, 4] - ic_car[:, 1] * ic_car[:, 3]
    ltot = np.linalg.norm((lx, ly, lz), axis=0)

    inc = np.degrees(np.arccos(lz / ltot))

    eccentricity = (df["r_apo"] - df["r_per"]) / (df["r_apo"] + df["r_per"])

    pot_file = POTENTIAL_FOLDER + POTENTIAL_TYPE
    pot = agama.Potential(pot_file)
    af = agama.ActionFinder(pot, interp=False)

    ioms = af(ic_car)
    jr = ioms[:, 0]
    jz = ioms[:, 1]
    jphi = ioms[:, 2]

    jtot = jr + np.abs(jphi) + jz

    ek = 0.5 * np.linalg.norm(ic_car[:, 3:], axis=1) ** 2
    ep = pot.potential(ic_car[:, :3])
    et = ep + ek

    r_circs = pot.Rcirc(E=et)
    xyz = np.column_stack((r_circs, r_circs * 0, r_circs * 0))
    v_circs = np.sqrt(-r_circs * pot.force(xyz)[:, 0])
    vel = np.column_stack((v_circs * 0, v_circs, v_circs * 0))
    init_conds = np.concatenate((xyz, vel), axis=1)
    lz_circ = af(init_conds)[:, 2]

    circ = lz / lz_circ

    check_sign = np.sign(ic_cyl[:, 4]) == np.sign(jphi)
    if len(np.where(~check_sign)[0]) > 0:
        raise ValueError("Vphi and Jphi of different signs.")

    orbits = agama.orbit(potential=pot, ic=ic_car, time=10 * pot.Tcirc(ic_car), trajsize=10000)
    max_z = []
    for orbit in orbits:
        max_z.append(np.max(np.abs(orbit[1][:, 2])))

    check_z = np.abs(ic_car[:, 2]) > np.array(max_z)
    if len(np.where(check_z)[0]) > 0:
        raise ValueError("Issues with z_max")


# df = orbit_param()
# get_orbits(df, POTENTIAL_TYPE = "PriceWhelan22.ini")
