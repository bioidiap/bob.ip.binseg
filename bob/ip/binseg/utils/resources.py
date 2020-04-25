#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Tools for interacting with the running computer or GPU"""

import os
import re
import subprocess
import shutil

import psutil

import logging

logger = logging.getLogger(__name__)

_nvidia_smi = shutil.which("nvidia-smi")
"""Location of the nvidia-smi program, if one exists"""

_nvidia_query = (
    # obtain possible values with ``nvidia-smi --help-query-gpu``
    "gpu_name",
    "memory.total",
    "memory.used",
    "utilization.gpu",
)
"""Query parameters for nvidia-smi"""

GB = float(2 ** 30)
"""The number of bytes in a gigabyte"""


def gpu_info(query=_nvidia_query):
    """Returns GPU information using nvidia-smi

    For a comprehensive list of options and help, execute ``nvidia-smi
    --help-query-gpu`` on a host with a GPU


    Parameters
    ----------

    query : list
        A list of query strings as defined by ``nvidia-smi --help-query-gpu``


    Returns
    -------

    data : tuple
        An ordered dictionary (organized as 2-tuples) containing the queried
        parameters.  If ``nvidia-smi`` is not available, returns a list of
        ``None`` objects.  Dots and underscores in the original NVIDIA naming
        convention are normalized with dashes.

  """

    if _nvidia_smi is not None:
        values = subprocess.getoutput(
            "%s --query-gpu=%s --format=csv,noheader"
            % (_nvidia_smi, ",".join(query))
        )
        values = [k.strip() for k in values.split(",")]
        regexp = re.compile(r"(\.|_)")
        fieldnames = [k.sub("-", k) for k in query]
        return tuple(zip(fieldnames, values))


_CLUSTER = []
"""List of processes currently being monitored"""


def cpu_info():
    """Returns process (+child) information using ``psutil``.

    This call examines the current process plus any spawn child and returns the
    combined resource usage summary for the process group.


    Returns
    -------

    data : tuple
        An ordered dictionary (organized as 2-tuples) containing these entries:

        0. ``system-memory-total`` (:py:class:`float`): total memory available,
           in gigabytes
        1. ``system-memory-used`` (:py:class:`float`): total memory used from
           the system, in gigabytes
        2. ``system-cpu-count`` (:py:class:`int`): number of logical CPUs
           available
        3. ``rss`` (:py:class:`float`):  RAM currently used by
           process and children, in gigabytes
        3. ``vms`` (:py:class:`float`):  total memory (RAM + swap) currently
           used by process and children, in gigabytes
        4. ``cpu-percent`` (:py:class:`float`): percentage of the total CPU
           used by this process and children (recursively) since last call
           (first time called should be ignored).  This number depends on the
           number of CPUs in the system and can be greater than 100%
        5. ``processes`` (:py:class:`int`): total number of processes including
           self and children (recursively)
        6. ``open-files`` (:py:class:`int`): total number of open files by
           self and children

    """

    global _CLUSTER
    if (not _CLUSTER) or (_CLUSTER[0] != psutil.Process()):  #initialization
        this = psutil.Process()
        _CLUSTER = [this] + this.children(recursive=True)
        # touch cpu_percent() at least once for all
        [k.cpu_percent(interval=None) for k in _CLUSTER]
    else:
        # check all cluster components and update process list
        # done so we can keep the cpu_percent() initialization
        children = _CLUSTER[0].children()
        stored_children = set(_CLUSTER[1:])
        current_children = set(_CLUSTER[0].children())
        keep_children = stored_children - current_children
        new_children = current_children - stored_children
        [k.cpu_percent(interval=None) for k in new_children]
        _CLUSTER = _CLUSTER[:1] + list(keep_children) + list(new_children)

    memory_info = [k.memory_info() for k in _CLUSTER]

    return (
        ("system-memory-total", psutil.virtual_memory().total / GB),
        ("system-memory-used", psutil.virtual_memory().used / GB),
        ("system-cpu-count", psutil.cpu_count(logical=True)),
        ("rss", sum([k.rss for k in memory_info]) / GB),
        ("vms", sum([k.vms for k in memory_info]) / GB),
        ("cpu-percent", sum(k.cpu_percent(interval=None) for k in _CLUSTER)),
        ("processes", len(_CLUSTER)),
        ("open-files", sum(len(k.open_files()) for k in _CLUSTER)),
    )
