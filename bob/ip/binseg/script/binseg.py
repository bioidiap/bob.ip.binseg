#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""The main entry for bob ip binseg (click-based) scripts."""

import os
import re
import sys
import time
import random
import tempfile
import urllib.request

import pkg_resources
import click
from click_plugins import with_plugins
from tqdm import tqdm

import numpy
import torch

from bob.extension.scripts.click_helper import AliasedGroup

import logging
logger = logging.getLogger(__name__)


def setup_pytorch_device(name):
    """Sets-up the pytorch device to use


    Parameters
    ----------

    name : str
        The device name (``cpu``, ``cuda:0``, ``cuda:1``, and so on).  If you
        set a specific cuda device such as ``cuda:1``, then we'll make sure it
        is currently set.


    Returns
    -------

    device : :py:class:`torch.device`
        The pytorch device to use, pre-configured (and checked)

    """

    if name.startswith("cuda:"):
        # In case one has multiple devices, we must first set the one
        # we would like to use so pytorch can find it.
        logger.info(f"User set device to '{name}' - trying to force device...")
        os.environ['CUDA_VISIBLE_DEVICES'] = name.split(":",1)[1]
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is not currently available, but " \
                    f"you set device to '{name}'")
        # Let pytorch auto-select from environment variable
        return torch.device("cuda")

    elif name.startswith("cuda"):  #use default device
        logger.info(f"User set device to '{name}' - using default CUDA device")
        assert os.environ.get('CUDA_VISIBLE_DEVICES') is not None

    #cuda or cpu
    return torch.device(name)


def set_seeds(value, all_gpus):
    """Sets up all relevant random seeds (numpy, python, cuda)

    If running with multiple GPUs **at the same time**, set ``all_gpus`` to
    ``True`` to force all GPU seeds to be initialized.

    Reference: `PyTorch page for reproducibility
    <https://pytorch.org/docs/stable/notes/randomness.html>`_.


    Parameters
    ----------

    value : int
        The random seed value to use

    all_gpus : :py:class:`bool`, Optional
        If set, then reset the seed on all GPUs available at once.  This is
        normally **not** what you want if running on a single GPU

    """

    random.seed(value)
    numpy.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)  #noop if cuda not available

    # set seeds for all gpus
    if all_gpus:
        torch.cuda.manual_seed_all(value)  #noop if cuda not available


def set_reproducible_cuda():
    """Turns-off all CUDA optimizations that would affect reproducibility

    For full reproducibility, also ensure not to use multiple (parallel) data
    lowers.  That is setup ``num_workers=0``.

    Reference: `PyTorch page for reproducibility
    <https://pytorch.org/docs/stable/notes/randomness.html>`_.


    """

    # ensure to use only optimization algos for cuda that are known to have
    # a deterministic effect (not random)
    torch.backends.cudnn.deterministic = True

    # turns off any optimization tricks
    torch.backends.cudnn.benchmark = False


def escape_name(v):
    """Escapes a name so it contains filesystem friendly characters only

    This function escapes every character that's not a letter, ``_``, ``-``,
    ``.`` or space with an ``-``.


    Parameters
    ==========

    v : str
        String to be escaped


    Returns
    =======

    s : str
        Escaped string

    """
    return re.sub(r'[^\w\-_\. ]', '-', v)


def save_sh_command(destfile):
    """Records command-line to reproduce this experiment

    This function can record the current command-line used to call the script
    being run.  It creates an executable ``bash`` script setting up the current
    working directory and activating a conda environment, if needed.  It
    records further information on the date and time the script was run and the
    version of the package.


    Parameters
    ----------

    destfile : str
        Path leading to the file where the commands to reproduce the current
        run will be recorded.  This file cannot be overwritten by this
        function.  If needed, you should check and remove an existing file
        **before** calling this function.

    """

    if os.path.exists(destfile) and not overwrite:
        logger.info(f"Not overwriting existing file '{destfile}'")
        return

    logger.info(f"Writing command-line for reproduction at '{destfile}'...")
    os.makedirs(os.path.dirname(destfile), exist_ok=True)

    with open(destfile, "wt") as f:
        f.write("#!/usr/bin/env sh\n")
        f.write(f"# date: {time.asctime()}\n")
        version = pkg_resources.require("bob.ip.binseg")[0].version
        f.write(f"# version: {version} (bob.ip.binseg)\n")
        f.write(f"# platform: {sys.platform}\n")
        f.write("\n")
        args = []
        for k in sys.argv:
            if " " in k:
                args.append(f'"{k}"')
            else:
                args.append(k)
        if os.environ.get("CONDA_DEFAULT_ENV") is not None:
            f.write(f"#conda activate {os.environ['CONDA_DEFAULT_ENV']}\n")
        f.write(f"#cd {os.path.realpath(os.curdir)}\n")
        f.write(" ".join(args) + "\n")
    os.chmod(destfile, 0o755)


def download_to_tempfile(url, progress=False):
    """Downloads a file to a temporary named file and returns it

    Parameters
    ----------

    url : str
        The URL pointing to the file to download

    progress : :py:class:`bool`, Optional
        If a progress bar should be displayed for downloading the URL.


    Returns
    -------

    f : tempfile.NamedTemporaryFile
        A named temporary file that contains the downloaded URL

    """

    file_size = 0
    response = urllib.request.urlopen(url)
    meta = response.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")

    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    progress &= bool(file_size)

    f = tempfile.NamedTemporaryFile()

    with tqdm(total=file_size, disable=not progress) as pbar:
        while True:
            buffer = response.read(8192)
            if len(buffer) == 0:
                break
            f.write(buffer)
            pbar.update(len(buffer))

    f.flush()
    f.seek(0)
    return f


@with_plugins(pkg_resources.iter_entry_points("bob.ip.binseg.cli"))
@click.group(cls=AliasedGroup)
def binseg():
    """Binary 2D Image Segmentation Benchmark commands."""
