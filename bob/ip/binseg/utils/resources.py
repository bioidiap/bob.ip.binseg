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

_nvidia_starter_query = (
    # obtain possible values with ``nvidia-smi --help-query-gpu``
    "gpu_name",
    "driver_version",
    "memory.total",
)
"""Query parameters for logging static GPU information"""

_nvidia_log_query = (
    # obtain possible values with ``nvidia-smi --help-query-gpu``
    "memory.used",
    "memory.free",
    "utilization.memory",
    "utilization.gpu",
)
"""Query parameters for logging performance of GPU"""

GB = float(2 ** 30)
"""The number of bytes in a gigabyte"""


def gpu_info(query=_nvidia_starter_query):
    """Returns GPU (static) information using nvidia-smi

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
        regexp = re.compile(r"(\.|-)")
        fieldnames = [regexp.sub("_", k) for k in query]
        return tuple(zip(fieldnames, values))


def gpu_log(query=_nvidia_log_query):
    """Returns GPU information about current non-static status using nvidia-smi

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
        convention are normalized with dashes.  Percentage information is left
        alone, memory information is transformed in to gigabytes.

  """

    if _nvidia_smi is not None:
        values = subprocess.getoutput(
            "%s --query-gpu=%s --format=csv,noheader"
            % (_nvidia_smi, ",".join(query))
        )
        values = [k.strip() for k in values.split(",")]
        t_values = []
        for k in values:
            if k.endswith('%'): t_values.append(k[:-1].strip())
            elif k.endswith('MiB'): t_values.append(float(k[:-3].strip())/1024)
        regexp = re.compile(r"(\.|-)")
        fieldnames = [regexp.sub("_", k) for k in query]
        return tuple(zip(fieldnames, values))


_CLUSTER = []
"""List of processes currently being monitored"""


def cpu_info():
    """Returns static CPU information about the current system.


    Returns
    -------

    data : tuple
        An ordered dictionary (organized as 2-tuples) containing these entries:

        0. ``cpu_memory_total`` (:py:class:`float`): total memory available,
           in gigabytes
        1. ``cpu_count`` (:py:class:`int`): number of logical CPUs available

    """

    return (
        ("cpu_memory_total", psutil.virtual_memory().total / GB),
        ("cpu_count", psutil.cpu_count(logical=True)),
    )


def cpu_log():
    """Returns process (+child) information using ``psutil``.

    This call examines the current process plus any spawn child and returns the
    combined resource usage summary for the process group.


    Returns
    -------

    data : tuple
        An ordered dictionary (organized as 2-tuples) containing these entries:

        0. ``cpu_memory_used`` (:py:class:`float`): total memory used from
           the system, in gigabytes
        1. ``cpu_rss`` (:py:class:`float`):  RAM currently used by
           process and children, in gigabytes
        2. ``cpu_vms`` (:py:class:`float`):  total memory (RAM + swap) currently
           used by process and children, in gigabytes
        3. ``cpu_percent`` (:py:class:`float`): percentage of the total CPU
           used by this process and children (recursively) since last call
           (first time called should be ignored).  This number depends on the
           number of CPUs in the system and can be greater than 100%
        4. ``cpu_processes`` (:py:class:`int`): total number of processes
           including self and children (recursively)
        5. ``cpu_open_files`` (:py:class:`int`): total number of open files by
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
        ("cpu_memory_used", psutil.virtual_memory().used / GB),
        ("cpu_rss", sum([k.rss for k in memory_info]) / GB),
        ("cpu_vms", sum([k.vms for k in memory_info]) / GB),
        ("cpu_percent", sum(k.cpu_percent(interval=None) for k in _CLUSTER)),
        ("cpu_processes", len(_CLUSTER)),
        ("cpu_open_files", sum(len(k.open_files()) for k in _CLUSTER)),
    )
