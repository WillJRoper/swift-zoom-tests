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


def carve_out_region(
    pos, masses, vels, region_rad, max_pos, boxsize, replicate
):
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
    # Calculate the old boxsize
    boxsize = boxsize / replicate

    # Shift positions to centre on max_pos
    pos -= max_pos
    pos = (pos + boxsize / 2) % boxsize

    # Mask out particles outside the region_rad from max_pos
    mask = np.linalg.norm(pos - (boxsize / 2), axis=1) <= region_rad
    new_pos = pos[mask]
    new_vels = vels[mask]
    new_masses = masses[mask]

    return new_pos, new_masses, new_vels, pos


def make_bkg_uniform(boxsize, bkg_ngrid, rho, new_masses, region_rad):
    """
    Generate a uniform background of particles.

    Args:
        boxsize (float): The size of the simulation box.
        bkg_ngrid (int): The number of background particles per dimension.
        rho (float): The mass density of the dark matter particles.
        new_masses (np.ndarray): The masses of the dark matter particles.
        region_rad (float): The radius of the region to carve out.

    Returns:
        np.ndarray: The positions of the background particles.
        np.ndarray: The masses of the background particles.
        np.ndarray: The velocities of the background particles.
    """
    # Compute the total mass needed for the background particles
    total_mass = (rho * boxsize**3 / 10**10) - np.sum(new_masses)

    # Add the background particles
    xx, yy, zz = np.meshgrid(
        np.linspace(0, boxsize, bkg_ngrid),
        np.linspace(0, boxsize, bkg_ngrid),
        np.linspace(0, boxsize, bkg_ngrid),
    )
    bkg_pos = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1)

    # Cut out the high resolution region
    mask = np.linalg.norm(bkg_pos - (boxsize / 2), axis=1) > region_rad
    bkg_pos = bkg_pos[mask]

    # Define background velocities and masses
    bkg_vels = np.zeros(bkg_pos.shape)
    bkg_masses = np.ones(bkg_pos.shape[0]) * (total_mass / bkg_pos.shape[0])

    return bkg_pos, bkg_masses, bkg_vels, boxsize


def make_bkg_gradient(boxsize, bkg_ngrid, rho, new_masses, region_rad):
    """
    Generate background particles with a number density gradient.

    Args:
        boxsize (float): The size of the simulation box.
        bkg_ngrid (int): The number of background particles per dimension.
        rho (float): The mass density of the dark matter particles.
        new_masses (np.ndarray): The masses of the dark matter particles.
        region_rad (float): The radius of the high resolution region.

    Returns:
        np.ndarray: The positions of the background particles.
        np.ndarray: The masses of the background particles.
        np.ndarray: The velocities of the background particles.
    """
    # Compute the total mass needed for the background particles
    total_mass = (rho * boxsize**3 / 10**10) - np.sum(new_masses)

    # Generate grids of background particles in shells to create a gradient
    grid_radius = region_rad * 2
    bkg_poss = []
    ngen = 0
    while True:
        print(
            "Generating background particles within "
            f"{boxsize /2 + grid_radius}. "
            f"Currently have {ngen}"
        )

        grid_pos, _, _, _ = make_bkg_uniform(
            boxsize / 2 + grid_radius,
            int(bkg_ngrid // (boxsize / (boxsize / 2 + grid_radius)) + 2),
            rho,
            new_masses,
            region_rad,
        )

        # Add some randomness
        grid_pos += np.random.uniform(-0.1, 0.1, grid_pos.shape)

        # Remove any particles inside the zoom region
        mask = np.linalg.norm(grid_pos - (boxsize / 2), axis=1) > region_rad
        grid_pos = grid_pos[mask]

        # Remove any particles outside the box
        mask = np.logical_or(grid_pos[:, 0] < 0, grid_pos[:, 1] < 0)
        mask = np.logical_or(mask, grid_pos[:, 2] < 0)
        mask = np.logical_or(mask, grid_pos[:, 0] > boxsize)
        mask = np.logical_or(mask, grid_pos[:, 1] > boxsize)
        mask = np.logical_or(mask, grid_pos[:, 2] > boxsize)
        grid_pos = grid_pos[~mask]

        # Add the grid to the list
        bkg_poss.append(grid_pos)
        ngen += grid_pos.shape[0]

        if boxsize / 2 + grid_radius > boxsize:
            break

        grid_radius *= 2

    print(f"Limiting to {bkg_ngrid**3} with a random selection")

    # Now choose bkg_ngrid**3 particles randomly
    bkg_pos = np.concatenate(bkg_poss)
    np.random.shuffle(bkg_pos)
    bkg_pos = bkg_pos[: bkg_ngrid**3, :]

    if bkg_pos.size < bkg_ngrid**3:
        raise ValueError("Not enough background particles generated.")

    # Shift the background particles to the centre of the box
    bkg_pos -= boxsize / 2
    bkg_pos = (bkg_pos + boxsize) % boxsize

    # Define background velocities
    bkg_vels = np.zeros((bkg_ngrid**3, 3))

    # Define background masses
    bkg_masses = np.ones(bkg_pos.shape[0])

    # Walk out in annuli scaling the mass
    radii = np.linspace(0, boxsize, 100)
    for i, r in enumerate(radii[:-1]):
        # Get all particles in this annulus
        mask = np.logical_and(
            np.linalg.norm(bkg_pos - (boxsize / 2), axis=1) >= r,
            np.linalg.norm(bkg_pos - (boxsize / 2), axis=1) < radii[i + 1],
        )

        if mask.sum() == 0:
            continue

        # Calculate the volume of this annulus accounting for how much of the
        # anullus is outside the simulation volume (note, this is simplified by
        # the fact the background particles car currently centred in the volume)
        vol = np.pi * (radii[i + 1] ** 2 - r**2) * (1 - (r / boxsize) ** 3)

        # Scale the mass of the particles in this annulus
        bkg_masses[mask] = rho * vol / mask.sum()

    # Finally, normalise and rescale the masses to make sure we have the right
    # total mass
    bkg_masses /= np.sum(bkg_masses)
    bkg_masses *= total_mass

    return bkg_pos, bkg_masses, bkg_vels, boxsize


def _downsample_box(positions, velocities, masses, target_num_particles):
    """
    Downsample the particle distribution to a target number of particles.

    By construction this will conserve mass but all background particles will
    have the same mass.

    Args:
        positions (np.ndarray): The positions of the background particles.
        velocities (np.ndarray): The velocities of the background particles.
        masses (np.ndarray): The masses of the background particles.
        target_num_particles (int): The target number of particles.

    Returns:
        np.ndarray: The downsampled positions of the background particles.
        np.ndarray: The downsampled velocities of the background particles.
        np.ndarray: The downsampled masses of the background particles.
    """
    if target_num_particles >= len(masses):
        raise ValueError(
            "Target number of particles must be less than "
            "the original number of particles."
        )

    # Normalize the masses to use as probabilities for sampling
    total_mass = np.sum(masses)
    probabilities = masses / total_mass

    # Randomly choose particles based on the probabilities
    selected_indices = np.random.choice(
        len(masses), size=target_num_particles, p=probabilities, replace=False
    )

    # Select the corresponding particles
    downsampled_positions = positions[selected_indices]
    downsampled_velocities = velocities[selected_indices]

    return downsampled_positions, downsampled_velocities


def make_bkg_downsampled(
    orig_positions,
    orig_velocities,
    orig_masses,
    bkg_ngrid,
    rho,
    boxsize,
    new_masses,
    replicate,
    cent,
    rad,
):
    """
    Downsample the particle distribution to a target number of particles.

    By construction this will conserve mass but all background particles will
    have the same mass.

    Args:
        positions (np.ndarray): The positions of the background particles.
        velocities (np.ndarray): The velocities of the background particles.
        masses (np.ndarray): The masses of the background particles.
        target_num_particles (int): The target number of particles.

    Returns:
        np.ndarray: The downsampled positions of the background particles.
        np.ndarray: The downsampled velocities of the background particles.
        np.ndarray: The downsampled masses of the background particles.
    """
    # Downsample the original box
    downsampled_pos, downsampled_vels = _downsample_box(
        orig_positions,
        orig_velocities,
        orig_masses,
        int((bkg_ngrid / replicate) ** 3),
    )

    # Carve out the zoom region
    mask = np.linalg.norm(downsampled_pos - cent, axis=1) <= rad
    bkg_pos = downsampled_pos[mask]
    bkg_vels = downsampled_vels[mask]

    # Replicate the downsampled positions and vels
    for i in range(replicate):
        for j in range(replicate):
            for k in range(replicate):
                if i == 0 and j == 0 and k == 0:
                    continue

                bkg_pos = np.concatenate(
                    (
                        bkg_pos,
                        downsampled_pos
                        + np.array([i * boxsize, j * boxsize, k * boxsize]),
                    )
                )
                bkg_vels = np.concatenate((bkg_vels, downsampled_vels))

    # Compute the total mass needed for the background particles
    total_mass = (rho * (boxsize * replicate) ** 3 / 10**10) - np.sum(
        new_masses
    )

    # Compute the mass of the background particles
    bkg_masses = np.ones(bkg_pos.shape[0]) * (total_mass / bkg_pos.shape[0])

    return bkg_pos, bkg_masses, bkg_vels


def write_ics(
    output_basename,
    new_pos,
    new_vels,
    new_masses,
    bkg_pos,
    bkg_vels,
    bkg_masses,
    boxsize,
    region_rad,
    bkg_ngrid,
    npart,
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
    # Create radius string
    r_str = str(region_rad).replace(".", "p")

    # Create simulation tag
    tag = (
        f"L{int(boxsize):04}N{int(npart**(1/3)):04}NBKG{bkg_ngrid:04}R{r_str}"
    )

    print(f"Writing {tag}")

    if uniform_bkg:
        output_file = f"ics/{output_basename}_{tag}_uniformbkg.hdf5"
    elif downsample:
        output_file = f"ics/{output_basename}_{tag}_downsampled.hdf5"
    else:
        output_file = f"ics/{output_basename}_{tag}.hdf5"

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
    ngrid,
    bkg_ngrid,
    region_rad,
    replicate,
    little_h=0.6777,
    uniform_bkg=True,
    downsample=False,
    omega_m=0.307,
    rho_crit=2.77536627 * 10**11,
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
    boxsize = meta.attrs["BoxSize"] / little_h

    # Read the dark matter coordinates, velocities and masses
    pos = hdf["PartType1"]["Coordinates"][...] / little_h
    masses = hdf["PartType1"]["Masses"][...] / little_h
    vels = hdf["PartType1"]["Velocities"][...]

    print(f"Loaded {pos.shape[0]} dark matter particles.")

    # Calculate the mass density
    rho = omega_m * rho_crit

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

    # Replicate the box if needed
    boxsize *= replicate
    bkg_ngrid *= replicate
    ngrid *= replicate

    print(f"Max pos: {max_pos}")

    # Carve out the high resolution region
    if subset:
        new_pos, new_masses, new_vels, pos = carve_out_region(
            pos,
            masses,
            vels,
            region_rad,
            max_pos,
            boxsize,
            replicate,
        )
    else:
        new_pos = pos
        new_masses = masses
        new_vels = vels

    print(f"Carved out {new_pos.shape[0]} particles.")

    # Centre the zoom region
    new_pos -= boxsize / replicate / 2
    new_pos = (new_pos + boxsize / 2) % boxsize

    # Make the background particles
    if uniform_bkg:
        bkg_pos, bkg_masses, bkg_vels, boxsize = make_bkg_uniform(
            boxsize, bkg_ngrid, rho, new_masses, region_rad
        )
    elif downsample:
        bkg_pos, bkg_masses, bkg_vels = make_bkg_downsampled(
            pos,
            vels,
            masses,
            bkg_ngrid,
            rho,
            boxsize / replicate,
            new_masses,
            replicate,
            max_pos,
            region_rad,
        )
    else:
        bkg_pos, bkg_masses, bkg_vels, boxsize = make_bkg_gradient(
            boxsize, bkg_ngrid, rho, new_masses, region_rad
        )

    print(f"Added {bkg_pos.shape[0]} background particles.")

    # Write the ICs
    write_ics(
        output_basename,
        new_pos,
        new_vels,
        new_masses,
        bkg_pos,
        bkg_vels,
        bkg_masses,
        boxsize,
        region_rad,
        bkg_ngrid,
        new_pos.shape[0],
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
        "--downsample",
        action="store_true",
        help="Whether to downsample the particles to make the background.",
        default=False,
    )
    parser.add_argument(
        "--uniform_bkg",
        action="store_true",
        help="Whether to use a uniform background.",
        default=False,
    )
    parser.add_argument(
        "--omega_m",
        type=float,
        help="The value of Omega_m.",
        default=0.307,
    )
    parser.add_argument(
        "--rho_crit",
        type=float,
        help="The value of the critical density in simulation units.",
        default=2.77536627 * 10**11,
    )

    args = parser.parse_args()

    region_rad = args.region_rad
    ngrid = args.ngrid
    bkg_ngrid = args.bkg_ngrid
    input_file = args.input_file
    out_basename = args.output_basename
    replicate = args.replicate
    downsample = args.downsample
    little_h = args.little_h
    uniform_bkg = args.uniform_bkg
    omega_m = args.omega_m
    rho_crit = args.rho_crit

    make_ics_dmo(
        input_file,
        out_basename,
        ngrid,
        bkg_ngrid,
        region_rad,
        replicate,
        little_h=little_h,
        uniform_bkg=uniform_bkg,
        downsample=downsample,
        omega_m=omega_m,
        rho_crit=rho_crit,
    )
