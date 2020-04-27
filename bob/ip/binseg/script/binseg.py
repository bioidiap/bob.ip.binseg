#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""The main entry for bob ip binseg (click-based) scripts."""

import os
import sys
import time
import tempfile
import urllib.request

import pkg_resources
import click
from click_plugins import with_plugins
from tqdm import tqdm

from bob.extension.scripts.click_helper import AliasedGroup

import logging
logger = logging.getLogger(__name__)


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
