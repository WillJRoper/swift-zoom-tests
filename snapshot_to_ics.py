#!/usr/bin/env python3
import h5py
import argparse
import numpy as np
from unyt import Mpc, km, s, Msun

from swiftsimio import Writer
from swiftsimio.units import cosmo_units


def write_ics(
    output_basename,
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
        output_basename (str): The base name of the output file.
        new_pos (np.ndarray): The positions of the dark matter particles.
        new_vels (np.ndarray): The velocities of the dark matter particles.
        new_masses (np.ndarray): The masses of the dark matter particles.
        bkg_pos (np.ndarray): The positions of the background particles.
        bkg_vels (np.ndarray): The velocities of the background particles.
        bkg_masses (np.ndarray): The masses of the background particles.
        boxsize (float): The size of the simulation box.
        region_rad (float): The radius of the high resolution region.
        bkg_ngrid (int): The number of background particles per dimension.
        npart (int): The number of dark matter particles.
    """
    # Create simulation tag
    tag = (
        f"L{int(boxsize[0]):04}N{int(new_pos.shape[0]**(1/3)):04}"
        f"NBKG{int(bkg_pos.shape[0]**(1/3)):04}"
    )

    output_file = f"ics/{output_basename}_{tag}.hdf5"

    print(cosmo_units)

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
    grp.create_dataset(
        "ParticleIDs",
        data=np.arange(0, bkg_masses.size, 1, dtype=int) + new_pos.shape[0],
        compression="gzip",
    )

    # Update the metadata
    num_part = hdf["Header"].attrs["NumPart_ThisFile"]
    num_part[2] = bkg_masses.size
    hdf["Header"].attrs["NumPart_ThisFile"] = num_part
    num_part = hdf["Header"].attrs["NumPart_Total"]
    num_part[2] = bkg_masses.size
    hdf["Header"].attrs["NumPart_Total"] = num_part
    mass_table = hdf["Header"].attrs["MassTable"]
    mass_table[2] = bkg_masses.mean()
    hdf["Header"].attrs["MassTable"] = mass_table

    hdf.close()


def make_ics_dmo(
    input_file,
    output_basename,
):
    """
    Generate DMO ics by carving out the densest region in the simulation box.

    Args:
        input_file (str): The path to the input file.
        output_basename (str): The path to the output file.
        ngrid (int): The number of grid cells along each dimension.
        bkg_ngrid (int): The number of background particles per dimension.
        region_rad (float): The radius of the region to carve out.
        replicate (int): The number of times to replicate the box.
        little_h (float): The value of little h.
        uniform_bkg (bool): Whether to use a uniform background.
        omega_m (float): The value of Omega_m.
        rho_crit (float): The value of the critical density in simulation units.
    """
    # Load the data
    hdf = h5py.File(input_file, "r")

    # Get the metadata
    meta = hdf["Header"]
    boxsize = meta.attrs["BoxSize"]

    # Read the dark matter coordinates, velocities and masses
    pos = hdf["PartType1"]["Coordinates"][...]
    masses = hdf["PartType1"]["Masses"][...]
    vels = hdf["PartType1"]["Velocities"][...]
    bkg_pos = hdf["PartType2"]["Coordinates"][...]
    bkg_masses = hdf["PartType2"]["Masses"][...]
    bkg_vels = hdf["PartType2"]["Velocities"][...]

    print(f"Loaded {pos.shape[0]} dark matter particles.")

    # Write the ICs
    write_ics(
        output_basename,
        pos,
        vels,
        masses,
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

    args = parser.parse_args()

    input_file = args.input_file
    out_basename = args.output_basename

    make_ics_dmo(
        input_file,
        out_basename,
    )
