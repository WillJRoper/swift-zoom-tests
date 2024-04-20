"""A script to generate DMO initial conditions.

This is done by carving out the densest region in the simulation box and
treating that as the high resolution region. The rest of the box is filled
with a uniform background of particles.

The box can be optionally replicated in each dimension to simulate different
zoom scenarios.

Example:
    $ python make_dmo_ics_uniform_bkg.py \
        --input_file /path/to/input.hdf5 \
        --output_basename output \
        --ngrid 32 \
        --bkg_ngrid 64 \
        --region_rad 5 \
        --replicate 1
"""
import h5py
from tqdm import tqdm
import argparse
import numpy as np
from unyt import Mpc, km, s, Msun

from swiftsimio import Writer
from swiftsimio.units import cosmo_units


def make_eagle_ics_dmo_uniform_bkg(
    input_file,
    output_file,
    ngrid,
    bkg_ngrid,
    region_rad,
    replicate,
    little_h=0.6777,
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
    """
    # Load the EAGLE data
    hdf = h5py.File(input_file, "r")

    # Get the metadata
    meta = hdf["Header"]
    boxsize = meta.attrs["BoxSize"] / little_h

    # Read the dark matter coordinates, velocities and masses
    pos = hdf["PartType1"]["Coordinates"][...] / little_h
    masses = hdf["PartType1"]["Masses"][...]
    vels = hdf["PartType1"]["Velocities"][...]

    print(f"Loaded {pos.shape[0]} dark matter particles.")

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

    # Mask out particles outside the region_rad from max_pos
    mask = np.linalg.norm(pos - max_pos, axis=1) < region_rad
    new_pos = pos[mask]
    new_vels = vels[mask]
    new_masses = masses[mask]

    print(f"Carved out {new_pos.shape[0]} particles.")

    # Replicate the box if needed
    boxsize *= replicate
    bkg_ngrid *= replicate

    # Add the background particles
    xx, yy, zz = np.meshgrid(
        np.linspace(0, boxsize, bkg_ngrid),
        np.linspace(0, boxsize, bkg_ngrid),
        np.linspace(0, boxsize, bkg_ngrid),
    )
    bkg_pos = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1)
    bkg_vels = np.zeros((bkg_ngrid**3, 3))
    bkg_masses = np.ones(bkg_ngrid**3) * (
        np.sum(masses[~mask]) / bkg_ngrid**3
    )

    print(f"Added {bkg_pos.shape[0]} background particles.")

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

    return grid, pos, bkg_pos


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

    args = parser.parse_args()

    region_rad = args.region_rad
    ngrid = args.ngrid
    bkg_ngrid = args.bkg_ngrid
    input_file = args.input_file
    replicate = args.replicate
    little_h = args.little_h

    out_file = (
        f"ics/{args.output_basename}_rad{region_rad}_"
        f"bkg{bkg_ngrid}_replicate{replicate}.hdf5"
    )

    grid, high_res_pos, bkg_pos = make_eagle_ics_dmo_uniform_bkg(
        input_file,
        out_file,
        ngrid,
        bkg_ngrid,
        region_rad,
        replicate,
        little_h=little_h,
    )
