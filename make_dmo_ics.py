"""A script to generate DMO initial conditions.

This is done by carving out the densest region in the simulation box and
treating that as the high resolution region. The rest of the box is filled
with a background of particles that increase in number density towards the
zoom region.

The box can be optionally replicated in each dimension to simulate different
zoom scenarios.

Example:
    $ python make_dmo_ics.py \
        --input_file /path/to/input.hdf5 \
        --output_basename output \
        --ngrid 32 \
        --bkg_ngrid 64 \
        --region_rad 5 \
        --replicate 1 \
        --little_h 0.6777 \
        --uniform_bkg
"""
import h5py
from tqdm import tqdm
import argparse
import numpy as np
from unyt import Mpc, km, s, Msun

from swiftsimio import Writer
from swiftsimio.units import cosmo_units


def get_max_pos(pos, masses, ngrid, boxsize):
    """
    Find the densest grid point in the simulation box.

    Args:
        pos (np.ndarray): The positions of the dark matter particles.
        masses (np.ndarray): The masses of the dark matter particles.
        ngrid (int): The number of grid cells along each dimension.
        boxsize (float): The size of the simulation box.

    Returns:
        np.ndarray: The position of the densest grid point.
    """
    # Grid the masses to find the densest grid point
    grid = np.zeros((ngrid, ngrid, ngrid))
    cell_width = boxsize / ngrid

    # Populate the grid with the dark matter particles
    for p in tqdm(range(masses.size)):
        # Compute the grid cell for this particle
        i = int(pos[p, 0] / cell_width)
        j = int(pos[p, 1] / cell_width)
        k = int(pos[p, 2] / cell_width)

        # Add the mass to the grid cell
        grid[i, j, k] += masses[p]

    # Find the position of the densest cell
    max_cell = np.unravel_index(np.argmax(grid), grid.shape)
    max_pos = (np.array(max_cell) * cell_width) + cell_width / 2

    return max_pos


def carve_out_region(pos, masses, vels, region_rad, max_pos, boxsize):
    """
    Carve out the high resolution region from the dark matter particles.

    Args:
        pos (np.ndarray): The positions of the dark matter particles.
        masses (np.ndarray): The masses of the dark matter particles.
        vels (np.ndarray): The velocities of the dark matter particles.
        region_rad (float): The radius of the region to carve out.
        max_pos (np.ndarray): The position of the high resolution region.
        boxsize (float): The size of the simulation box.

    Returns:
        np.ndarray: The new positions of the dark matter particles.
        np.ndarray: The new masses of the dark matter particles.
        np.ndarray: The new velocities of the dark matter particles.
    """
    # Shift positions to centre on max_pos
    pos -= max_pos
    pos += boxsize
    pos %= boxsize

    # Mask out particles outside the region_rad from max_pos
    mask = np.linalg.norm(pos - (boxsize / 2), axis=1) < region_rad
    new_pos = pos[mask]
    new_vels = vels[mask]
    new_masses = masses[mask]

    return new_pos, new_masses, new_vels


def make_bkg_uniform(boxsize, bkg_ngrid, replicate, rho, new_masses):
    """
    Generate a uniform background of particles.

    Args:
        boxsize (float): The size of the simulation box.
        bkg_ngrid (int): The number of background particles per dimension.
        replicate (int): The number of times to replicate the box.
        rho (float): The mass density of the dark matter particles.
        new_masses (np.ndarray): The masses of the dark matter particles.

    Returns:
        np.ndarray: The positions of the background particles.
        np.ndarray: The masses of the background particles.
        np.ndarray: The velocities of the background particles.
    """
    # Replicate the box if needed
    new_boxsize = boxsize * replicate
    bkg_ngrid *= replicate

    # Compute the total mass needed for the background particles
    total_mass = rho * boxsize**3 - np.sum(new_masses)

    # Add the background particles
    xx, yy, zz = np.meshgrid(
        np.linspace(0, boxsize, bkg_ngrid),
        np.linspace(0, boxsize, bkg_ngrid),
        np.linspace(0, boxsize, bkg_ngrid),
    )
    bkg_pos = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1)

    # Cut out the high resolution region
    mask = np.linalg.norm(bkg_pos - (boxsize / 2), axis=1) < boxsize / 2
    bkg_pos = bkg_pos[mask]

    # Define background velocities and masses
    bkg_vels = np.zeros((bkg_ngrid**3, 3))
    bkg_masses = np.ones(bkg_ngrid**3) * (total_mass / bkg_pos.shape[0] ** 3)

    return bkg_pos, bkg_masses, bkg_vels, new_boxsize


def make_bkg_gradient(
    boxsize, bkg_ngrid, replicate, rho, new_masses, region_rad
):
    """
    Generate background particles with a number density gradient.

    Args:
        boxsize (float): The size of the simulation box.
        bkg_ngrid (int): The number of background particles per dimension.
        replicate (int): The number of times to replicate the box.
        rho (float): The mass density of the dark matter particles.
        new_masses (np.ndarray): The masses of the dark matter particles.
        region_rad (float): The radius of the high resolution region.

    Returns:
        np.ndarray: The positions of the background particles.
        np.ndarray: The masses of the background particles.
        np.ndarray: The velocities of the background particles.
    """
    # Replicate the box if needed
    new_boxsize = boxsize * replicate
    bkg_ngrid *= replicate

    # Compute the total mass needed for the background particles
    total_mass = rho * new_boxsize**3 - np.sum(new_masses)

    # Generate grids of background particles in shells to create a gradient
    grid_radius = region_rad * 2
    bkg_poss = []
    while grid_radius < boxsize:
        print("Generating background particles within:", grid_radius)

        grid_pos, _, _, _ = make_bkg_uniform(
            boxsize / 2 + grid_radius,
            bkg_ngrid // replicate,
            1,
            rho,
            new_masses,
        )

        # Add some randomness
        grid_pos += np.random.uniform(-0.1, 0.1, grid_pos.shape)

        # Remove any particles inside the zoom region
        mask = np.linalg.norm(grid_pos - (boxsize / 2), axis=1) < region_rad
        grid_pos = grid_pos[~mask]

        # Remove any particles outside the box
        mask = np.logical_or(grid_pos[:, 0] < 0, grid_pos[:, 1] < 0)
        mask = np.logical_or(mask, grid_pos[:, 2] < 0)
        mask = np.logical_or(mask, grid_pos[:, 0] > new_boxsize)
        mask = np.logical_or(mask, grid_pos[:, 1] > new_boxsize)
        mask = np.logical_or(mask, grid_pos[:, 2] > new_boxsize)
        grid_pos = grid_pos[~mask]

        # Add the grid to the list
        bkg_poss.append(grid_pos)

        grid_radius *= 2

    # Now choose bkg_ngrid**3 particles randomly
    bkg_pos = np.concatenate(bkg_poss)
    np.random.shuffle(bkg_pos)
    bkg_pos = bkg_pos[: bkg_ngrid**3, :]

    # Define background velocities
    bkg_vels = np.zeros((bkg_ngrid**3, 3))

    # Define background masses
    bkg_masses = np.ones(bkg_ngrid**3) * (total_mass / bkg_ngrid**3)

    # Define distances of particles ready to scale the masses
    dist = np.linalg.norm(bkg_pos - (boxsize / 2), axis=1)

    # Adjust the mass to keep mass density constant
    # Since more particles will be closer to max_pos, reduce their mass
    # inversely proportional to increased number density
    mass_scale_factors = np.exp(-dist / (boxsize / bkg_ngrid))
    bkg_masses *= mass_scale_factors

    return bkg_pos, bkg_masses, bkg_vels, new_boxsize


def write_ics(
    output_file,
    new_pos,
    new_vels,
    new_masses,
    bkg_pos,
    bkg_vels,
    bkg_masses,
    boxsize,
):
    """
    Write the initial conditions to a file.

    Args:
        output_file (str): The path to the output file.
        new_pos (np.ndarray): The positions of the dark matter particles.
        new_vels (np.ndarray): The velocities of the dark matter particles.
        new_masses (np.ndarray): The masses of the dark matter particles.
        bkg_pos (np.ndarray): The positions of the background particles.
        bkg_vels (np.ndarray): The velocities of the background particles.
        bkg_masses (np.ndarray): The masses of the background particles.
        boxsize (float): The size of the simulation box.
    """
    # Set up the IC writer
    ics = Writer(
        cosmo_units,
        np.array((boxsize, boxsize, boxsize)) * Mpc,
        dimension=3,
    )

    # Write the dark matter particles
    ics.dark_matter.coordinates = new_pos * Mpc
    ics.dark_matter.velocities = new_vels * km / s
    ics.dark_matter.masses = new_masses * 10**10 * Msun

    # Write the ICs
    ics.write(output_file)

    # Write the background separately
    hdf = h5py.File(output_file, "r+")
    grp = hdf.create_group("PartType2")
    grp.create_dataset("Coordinates", data=bkg_pos, compression="gzip")
    grp.create_dataset("Velocities", data=bkg_vels, compression="gzip")
    grp.create_dataset("Masses", data=bkg_masses, compression="gzip")

    # Update the metadata
    hdf["Header"].attrs["NumPart_ThisFile"][2] = bkg_masses.size
    hdf["Header"].attrs["NumPart_Total"][2] = bkg_masses.size
    hdf["Header"].attrs["NumPart_Total_HighWord"][2] = 0
    hdf["Header"].attrs["MassTable"][2] = bkg_masses[0]

    hdf.close()


def make_ics_dmo(
    input_file,
    output_file,
    ngrid,
    bkg_ngrid,
    region_rad,
    replicate,
    little_h=0.6777,
    uniform_bkg=True,
):
    """
    Generate DMO ics by carving out the densest region in the simulation box.

    Args:
        input_file (str): The path to the input file.
        output_file (str): The path to the output file.
        ngrid (int): The number of grid cells along each dimension.
        bkg_ngrid (int): The number of background particles per dimension.
        region_rad (float): The radius of the region to carve out.
        replicate (int): The number of times to replicate the box.
        little_h (float): The value of little h.
        uniform_bkg (bool): Whether to use a uniform background.
    """
    # Load the data
    hdf = h5py.File(input_file, "r")

    # Get the metadata
    meta = hdf["Header"]
    boxsize = meta.attrs["BoxSize"] / little_h

    # Read the dark matter coordinates, velocities and masses
    pos = hdf["PartType1"]["Coordinates"][...] / little_h
    masses = hdf["PartType1"]["Masses"][...]
    vels = hdf["PartType1"]["Velocities"][...]

    print(f"Loaded {pos.shape[0]} dark matter particles.")

    # Calculate the mass density
    rho = np.sum(masses) / boxsize**3

    # Do we need to select a subset or just use the whole box?
    subset = region_rad < boxsize / 2

    if not subset:
        print(
            f"WARNING: region_rad ({region_rad}) is "
            f"bigger than the box ({boxsize})."
            "The region will be limited to the entire box."
        )

    # Get the position of the peak density (or the centre of the box if
    # region_rad exncompasses the entire box).
    if subset:
        max_pos = get_max_pos(pos, masses, ngrid, boxsize)
    else:
        max_pos = np.array([boxsize / 2, boxsize / 2, boxsize / 2])

    print(f"Max pos: {max_pos}")

    # Carve out the high resolution region
    if subset:
        new_pos, new_masses, new_vels = carve_out_region(
            pos,
            masses,
            vels,
            region_rad,
            max_pos,
            boxsize,
        )
    else:
        new_pos = pos
        new_masses = masses
        new_vels = vels

    print(f"Carved out {new_pos.shape[0]} particles.")

    # Make the background particles
    if uniform_bkg:
        bkg_pos, bkg_masses, bkg_vels, boxsize = make_bkg_uniform(
            boxsize,
            bkg_ngrid,
            replicate,
            rho,
            new_masses,
        )
    else:
        bkg_pos, bkg_masses, bkg_vels, boxsize = make_bkg_gradient(
            boxsize,
            bkg_ngrid,
            replicate,
            rho,
            new_masses,
            region_rad,
        )

    print(f"Added {bkg_pos.shape[0]} background particles.")

    # Write the ICs
    write_ics(
        output_file,
        new_pos,
        new_vels,
        new_masses,
        bkg_pos,
        bkg_vels,
        bkg_masses,
        boxsize,
    )


if __name__ == "__main__":
    # Set up the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="The path to the input file.",
    )
    parser.add_argument(
        "--output_basename",
        type=str,
        help="The path to the output file.",
    )
    parser.add_argument(
        "--ngrid",
        type=int,
        help="The number of grid cells along each dimension.",
        default=32,
    )
    parser.add_argument(
        "--bkg_ngrid",
        type=int,
        help="The number of background particles per dimension.",
        default=64,
    )
    parser.add_argument(
        "--region_rad",
        type=float,
        help="The radius of the region to carve out.",
        default=5,
    )
    parser.add_argument(
        "--replicate",
        type=int,
        help="The number of times to replicate the box.",
        default=1,
    )
    parser.add_argument(
        "--little_h",
        type=float,
        help="The value of little h.",
        default=0.6777,
    )
    parser.add_argument(
        "--uniform_bkg",
        action="store_true",
        help="Whether to use a uniform background.",
        default=False,
    )

    args = parser.parse_args()

    region_rad = args.region_rad
    ngrid = args.ngrid
    bkg_ngrid = args.bkg_ngrid
    input_file = args.input_file
    replicate = args.replicate
    little_h = args.little_h
    uniform_bkg = args.uniform_bkg

    if uniform_bkg:
        out_file = (
            f"ics/{args.output_basename}_rad{region_rad}_"
            f"bkg{bkg_ngrid}_replicate{replicate}_uniformbkg.hdf5"
        )
    else:
        out_file = (
            f"ics/{args.output_basename}_rad{region_rad}_"
            f"bkg{bkg_ngrid}_replicate{replicate}.hdf5"
        )

    make_ics_dmo(
        input_file,
        out_file,
        ngrid,
        bkg_ngrid,
        region_rad,
        replicate,
        little_h=little_h,
        uniform_bkg=uniform_bkg,
    )