"""A test for snapshots generated by SWIFT with a zoom region.

Example:
    $ python zoom_snap_test.py zoom_snap.hdf5
"""
import argparse
import h5py


def main():
    """Run test to ensure cell look up table works with zooms."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test for SWIFT zoom snapshots"
    )
    parser.add_argument("filename", help="Name of the file to test")

    args = parser.parse_args()

    # Open the file
    hdf = h5py.File(args.filename, "r")

    # Get the number of high res dark matter particles
    nr_high_res = hdf["Header"].attrs["NumPart_Total"][1]

    # Check for the presence of the zoom flag
    assert hdf["Header"].attrs["ZoomIn"] == 1, "Zoom flag not set"

    # Extract the cell metadata
    cell_meta = hdf["Cells"]["Metadata"]
    origin = cell_meta.attrs["Origin"]
    region_cdim = cell_meta.attrs["dimension"]
    nr_cells = cell_meta.attrs["nr_cells"]
    cell_width = cell_meta.attrs["size"]

    # Get the cell particle offsets and counts
    parttype1_offset = hdf["Cells"]["OffsetsInFile"]["PartType1"][...]
    parttype1_count = hdf["Cells"]["Counts"]["PartType1"][...]

    # Count how many particles we extract
    part1_count = 0

    # Loop over the cells
    for i in range(nr_cells):
        # Get the particle offset and count
        offset = parttype1_offset[i]
        count = parttype1_count[i]

        # Extract the particle data
        part1 = hdf["PartType1"]["Coordinates"][offset : offset + count, :]

        # Increment the particle count
        part1_count += count

        # Ensure all the extracted particles are in this cell
        assert (
            part1[:, 0] >= origin[0]
        ).all(), "Particle outside cell on x-axis"
        assert (
            part1[:, 0] <= origin[0] + region_cdim[0] * cell_width[0]
        ).all(), "Particle outside cell on x-axis"
        assert (
            part1[:, 1] >= origin[1]
        ).all(), "Particle outside cell on y-axis"
        assert (
            part1[:, 1] <= origin[1] + region_cdim[1] * cell_width[1]
        ).all(), "Particle outside cell on y-axis"
        assert (
            part1[:, 2] >= origin[2]
        ).all(), "Particle outside cell on z-axis"
        assert (
            part1[:, 2] <= origin[2] + region_cdim[2] * cell_width[2]
        ).all(), "Particle outside cell on z-axis"

    # Check that we have extracted all the particles
    assert part1_count == nr_high_res, "Not all particles extracted"
