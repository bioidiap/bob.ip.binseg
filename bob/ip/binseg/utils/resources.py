#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Tools for interacting with the running computer or GPU"""

import os
import subprocess
import shutil

import psutil

import logging

logger = logging.getLogger(__name__)

_nvidia_smi = shutil.which("nvidia-smi")
"""Location of the nvidia-smi program, if one exists"""


GB = float(2 ** 30)
"""The number of bytes in a gigabyte"""


def run_nvidia_smi(query, rename=None):
    """Returns GPU information from query

    For a comprehensive list of options and help, execute ``nvidia-smi
    --help-query-gpu`` on a host with a GPU


    Parameters
    ----------

    query : list
        A list of query strings as defined by ``nvidia-smi --help-query-gpu``

    rename : :py:class:`list`, Optional
        A list of keys to yield in the return value for each entry above.  It
        gives you the opportunity to rewrite some key names for convenience.
        This list, if provided, must be of the same length as ``query``.


    Returns
    -------

    data : :py:class:`tuple`, None
        An ordered dictionary (organized as 2-tuples) containing the queried
        parameters (``rename`` versions).  If ``nvidia-smi`` is not available,
        returns ``None``.  Percentage information is left alone,
        memory information is transformed to gigabytes (floating-point).

    """

    if _nvidia_smi is not None:

        if rename is None:
            rename = query
        else:
            assert len(rename) == len(query)

        values = subprocess.getoutput(
            "%s --query-gpu=%s --format=csv,noheader"
            % (_nvidia_smi, ",".join(query))
        )
        values = [k.strip() for k in values.split(",")]
        t_values = []
        for k in values:
            if k.endswith("%"):
                t_values.append(float(k[:-1].strip()))
            elif k.endswith("MiB"):
                t_values.append(float(k[:-3].strip()) / 1024)
            else:
                t_values.append(k)  #unchanged
        return tuple(zip(rename, t_values))


def gpu_constants():
    """Returns GPU (static) information using nvidia-smi

    See :py:func:`run_nvidia_smi` for operational details.

    Returns
    -------

    data : :py:class:`tuple`, None
        If ``nvidia-smi`` is not available, returns ``None``, otherwise, we
        return an ordered dictionary (organized as 2-tuples) containing the
        following ``nvidia-smi`` query information:

        * ``gpu_name``, as ``gpu_name`` (:py:class:`str`)
        * ``driver_version``, as ``gpu_driver_version`` (:py:class:`str`)
        * ``memory.total``, as ``gpu_memory_total`` (transformed to gigabytes,
          :py:class:`float`)

    """

    return run_nvidia_smi(
        ("gpu_name", "driver_version", "memory.total"),
        ("gpu_name", "gpu_driver_version", "gpu_memory_total"),
    )


def gpu_log():
    """Returns GPU information about current non-static status using nvidia-smi

    See :py:func:`run_nvidia_smi` for operational details.

    Returns
    -------

    data : :py:class:`tuple`, None
        If ``nvidia-smi`` is not available, returns ``None``, otherwise, we
        return an ordered dictionary (organized as 2-tuples) containing the
        following ``nvidia-smi`` query information:

        * ``memory.used``, as ``gpu_memory_used`` (transformed to gigabytes,
          :py:class:`float`)
        * ``memory.free``, as ``gpu_memory_free`` (transformed to gigabytes,
          :py:class:`float`)
        * ``utilization.memory``, as ``gpu_memory_percent``,
          (:py:class:`float`, in percent)
        * ``utilization.gpu``, as ``gpu_utilization``,
          (:py:class:`float`, in percent)

    """

    return run_nvidia_smi(
        ("memory.used", "memory.free", "utilization.memory", "utilization.gpu"),
        (
            "gpu_memory_used",
            "gpu_memory_free",
            "gpu_memory_percent",
            "gpu_percent",
        ),
    )


_CLUSTER = []
"""List of processes currently being monitored"""


def cpu_constants():
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
    if (not _CLUSTER) or (_CLUSTER[0] != psutil.Process()):  # initialization
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
