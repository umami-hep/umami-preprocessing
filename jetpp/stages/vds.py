import glob
import logging as log

import h5py

from jetpp.logger import setup_logger


def get_dtype(fname, group):
    return h5py.File(fname)[group].dtype


def get_n_per_jet(fname, group):
    shape = h5py.File(fname)[group].shape
    return shape[1] if len(shape) == 2 else None


def get_virtual_layout(fnames, group):
    # get sources
    sources = []
    total_length = 0
    for fname in fnames:
        with h5py.File(fname) as f:
            vsource = h5py.VirtualSource(f[group])
            total_length += vsource.shape[0]
            sources.append(vsource)

    # define layout of the vds
    dtype = get_dtype(fnames[0], group)
    n_per_jet = get_n_per_jet(fname, group)
    shape = (total_length,) if n_per_jet is None else (total_length, n_per_jet)
    layout = h5py.VirtualLayout(shape=shape, dtype=dtype)

    # fill the vds
    offset = 0
    for vsource in sources:
        length = vsource.shape[0]
        layout[offset : offset + length] = vsource
        offset += length

    return layout


def create_vritual_file(fname_pattern, vds_path):
    # get list of filenames
    fnames = glob.glob(str(fname_pattern))
    if not fnames:
        raise FileNotFoundError(f"No files matched pattern {fname_pattern}")

        # create virtual file
    vds_path.parent.mkdir(exist_ok=True)
    with h5py.File(vds_path, "w") as f:
        for group in h5py.File(fnames[0]):
            layout = get_virtual_layout(fnames, group)
            f.create_virtual_dataset(group, layout)


def main(config=None):
    setup_logger()

    title = " Creating virtual datasets "
    log.info(f"[bold green]{title:-^100}")
    for sample in config.components.samples:
        log.info(f"Creating virtual dataset for {sample}")
        log.info(f"[magenta]{sample.path} \n-> {sample.vds_path}")
        create_vritual_file(sample.path, sample.vds_path)
