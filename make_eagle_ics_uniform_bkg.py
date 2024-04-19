import argparse
import numpy as np

from swiftsimio import load
from swiftsimio import Writer
from swiftsimio.units import cosmo_units
import swiftsimio.metadata.particle as swp

# Add background particles to swiftsimio
swp.particle_name_underscores[6] = "dark_matter_bkg"
swp.particle_name_class[6] = "PartType2"
swp.particle_name_text[6] = "PartType2"


def make_eagle_ics_dmo_uniform_bkg(
    input_file,
    output_file,
    ngrid,
    bkg_ngrid,
    region_rad,
):
    """
    Generate DMO ics by carving out the densest region in the simulation box.

    Args:
        input_file (str): The path to the input file.
        output_file (str): The path to the output file.
        ngrid (int): The number of grid cells along each dimension.
        bkg_ngrid (int): The number of background particles per dimension.
        region_rad (float): The radius of the region to carve out.
    """
    # Load the EAGLE data
    data = load(input_file)

    # Get the metadata
    meta = data.metadata
    boxsize = meta.boxsize
    print(meta.field_names)

    # Read the dark matter coordinates, velocities and masses
    pos = data.dark_matter.coordinates
    masses = data.dark_matter.masses
    vels = data.dark_matter.velocities

    # Grid the masses to find the densest grid point
    grid = np.zeros((ngrid, ngrid, ngrid))
    cell_width = boxsize[0] / ngrid

    # Populate the grid with the dark matter particles
    for p in range(masses.size):
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

    # Set up the IC writer
    ics = Writer(cosmo_units, boxsize, dimension=3)

    # Write the dark matter particles
    ics.dark_matter.coordinates = new_pos
    ics.dark_matter.velocities = new_vels
    ics.dark_matter.masses = new_masses

    # Add the background particles
    xx, yy, zz = np.meshgrid(
        np.linspace(0, boxsize[0], bkg_ngrid),
        np.linspace(0, boxsize[1], bkg_ngrid),
        np.linspace(0, boxsize[2], bkg_ngrid),
    )
    bkg_pos = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1)
    bkg_vels = np.zeros((bkg_ngrid**3, 3))
    bkg_masses = np.ones(bkg_ngrid**3) * (
        np.sum(masses[~mask]) / bkg_ngrid**3
    )

    ics.dark_matter_bkg.coordinates = bkg_pos
    ics.dark_matter_bkg.velocities = bkg_vels
    ics.dark_matter_bkg.masses = bkg_masses

    # Write the ICs
    ics.write(output_file)

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

    args = parser.parse_args()

    region_rad = args.region_rad
    ngrid = args.ngrid
    bkg_ngrid = args.bkg_ngrid
    input_file = args.input_file

    out_file = (
        f"ics/{args.output_basename}_rad{region_rad}_bkg{bkg_ngrid}.hdf5"
    )

    grid, high_res_pos, bkg_pos = make_eagle_ics_dmo_uniform_bkg(
        input_file,
        out_file,
        ngrid,
        bkg_ngrid,
        region_rad,
    )
