# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tools for interacting with the running computer or GPU."""

import logging
import multiprocessing
import os
import queue
import shutil
import subprocess
import time

import numpy
import psutil

logger = logging.getLogger(__name__)

_nvidia_smi = shutil.which("nvidia-smi")
"""Location of the nvidia-smi program, if one exists."""


GB = float(2**30)
"""The number of bytes in a gigabyte."""


def run_nvidia_smi(query, rename=None):
    """Returns GPU information from query.

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

        # Get GPU information based on GPU ID.
        values = subprocess.getoutput(
            "%s --query-gpu=%s --format=csv,noheader --id=%s"
            % (
                _nvidia_smi,
                ",".join(query),
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
        )
        values = [k.strip() for k in values.split(",")]
        t_values = []
        for k in values:
            if k.endswith("%"):
                t_values.append(float(k[:-1].strip()))
            elif k.endswith("MiB"):
                t_values.append(float(k[:-3].strip()) / 1024)
            else:
                t_values.append(k)  # unchanged
        return tuple(zip(rename, t_values))


def gpu_constants():
    """Returns GPU (static) information using nvidia-smi.

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
    """Returns GPU information about current non-static status using nvidia-
    smi.

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
        * ``100*memory.used/memory.total``, as ``gpu_memory_percent``,
          (:py:class:`float`, in percent)
        * ``utilization.gpu``, as ``gpu_percent``,
          (:py:class:`float`, in percent)
    """

    retval = run_nvidia_smi(
        (
            "memory.total",
            "memory.used",
            "memory.free",
            "utilization.gpu",
        ),
        (
            "gpu_memory_total",
            "gpu_memory_used",
            "gpu_memory_free",
            "gpu_percent",
        ),
    )

    # re-compose the output to generate expected values
    return (
        retval[1],  # gpu_memory_used
        retval[2],  # gpu_memory_free
        ("gpu_memory_percent", 100 * (retval[1][1] / retval[0][1])),
        retval[3],  # gpu_percent
    )


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


class CPULogger:
    """Logs CPU information using :py:mod:`psutil`

    Parameters
    ----------

    pid : :py:class:`int`, Optional
        Process identifier of the main process (parent process) to observe
    """

    def __init__(self, pid=None):
        this = psutil.Process(pid=pid)
        self.cluster = [this] + this.children(recursive=True)
        # touch cpu_percent() at least once for all processes in the cluster
        [k.cpu_percent(interval=None) for k in self.cluster]

    def log(self):
        """Returns current process cluster information.

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

        # check all cluster components and update process list
        # done so we can keep the cpu_percent() initialization
        stored_children = set(self.cluster[1:])
        current_children = set(self.cluster[0].children(recursive=True))
        keep_children = stored_children - current_children
        new_children = current_children - stored_children
        gone = set()
        for k in new_children:
            try:
                k.cpu_percent(interval=None)
            except (psutil.ZombieProcess, psutil.NoSuchProcess):
                # child process is gone meanwhile
                # update the intermediate list for this time
                gone.add(k)
        new_children = new_children - gone
        self.cluster = (
            self.cluster[:1] + list(keep_children) + list(new_children)
        )

        memory_info = []
        cpu_percent = []
        open_files = []
        gone = set()
        for k in self.cluster:
            try:
                memory_info.append(k.memory_info())
                cpu_percent.append(k.cpu_percent(interval=None))
                open_files.append(len(k.open_files()))
            except (psutil.ZombieProcess, psutil.NoSuchProcess):
                # child process is gone meanwhile, just ignore it
                # it is too late to update any intermediate list
                # at this point, but ensures to update counts later on
                gone.add(k)

        return (
            ("cpu_memory_used", psutil.virtual_memory().used / GB),
            ("cpu_rss", sum([k.rss for k in memory_info]) / GB),
            ("cpu_vms", sum([k.vms for k in memory_info]) / GB),
            ("cpu_percent", sum(cpu_percent)),
            ("cpu_processes", len(self.cluster) - len(gone)),
            ("cpu_open_files", sum(open_files)),
        )


class _InformationGatherer:
    """A container to store monitoring information.

    Parameters
    ----------

    has_gpu : bool
        A flag indicating if we have a GPU installed on the platform or not

    main_pid : int
        The main process identifier to monitor

    logger : logging.Logger
        A logger to be used for logging messages
    """

    def __init__(self, has_gpu, main_pid, logger):
        self.cpu_logger = CPULogger(main_pid)
        self.keys = [k[0] for k in self.cpu_logger.log()]
        self.cpu_keys_len = len(self.keys)
        self.has_gpu = has_gpu
        self.logger = logger
        if self.has_gpu:
            self.keys += [k[0] for k in gpu_log()]
        self.data = [[] for _ in self.keys]

    def acc(self):
        """Accumulates another measurement."""
        for i, k in enumerate(self.cpu_logger.log()):
            self.data[i].append(k[1])
        if self.has_gpu:
            for i, k in enumerate(gpu_log()):
                self.data[i + self.cpu_keys_len].append(k[1])

    def summary(self):
        """Returns the current data."""

        if len(self.data[0]) == 0:
            self.logger.error("CPU/GPU logger was not able to collect any data")
        retval = []
        for k, values in zip(self.keys, self.data):
            retval.append((k, values))
        return tuple(retval)


def _monitor_worker(interval, has_gpu, main_pid, stop, queue, logging_level):
    """A monitoring worker that measures resources and returns lists.

    Parameters
    ==========

    interval : int, float
        Number of seconds to wait between each measurement (maybe a floating
        point number as accepted by :py:func:`time.sleep`)

    has_gpu : bool
        A flag indicating if we have a GPU installed on the platform or not

    main_pid : int
        The main process identifier to monitor

    stop : :py:class:`multiprocessing.Event`
        Indicates if we should continue running or stop

    queue : :py:class:`queue.Queue`
        A queue, to send monitoring information back to the spawner

    logging_level: int
        The logging level to use for logging from launched processes
    """

    logger = multiprocessing.log_to_stderr(level=logging_level)
    ra = _InformationGatherer(has_gpu, main_pid, logger)

    while not stop.is_set():
        try:
            ra.acc()  # guarantees at least an entry will be available
            time.sleep(interval)
        except Exception:
            logger.warning(
                "Iterative CPU/GPU logging did not work properly " "this once",
                exc_info=True,
            )
            time.sleep(0.5)  # wait half a second, and try again!

    queue.put(ra.summary())


class ResourceMonitor:
    """An external, non-blocking CPU/GPU resource monitor.

    Parameters
    ----------

    interval : int, float
        Number of seconds to wait between each measurement (maybe a floating
        point number as accepted by :py:func:`time.sleep`)

    has_gpu : bool
        A flag indicating if we have a GPU installed on the platform or not

    main_pid : int
        The main process identifier to monitor

    logging_level: int
        The logging level to use for logging from launched processes
    """

    def __init__(self, interval, has_gpu, main_pid, logging_level):
        self.interval = interval
        self.has_gpu = has_gpu
        self.main_pid = main_pid
        self.event = multiprocessing.Event()
        self.q = multiprocessing.Queue()
        self.logging_level = logging_level

        self.monitor = multiprocessing.Process(
            target=_monitor_worker,
            name="ResourceMonitorProcess",
            args=(
                self.interval,
                self.has_gpu,
                self.main_pid,
                self.event,
                self.q,
                self.logging_level,
            ),
        )

        self.data = None

    @staticmethod
    def monitored_keys(has_gpu):
        return _InformationGatherer(has_gpu, None, logger).keys

    def __enter__(self):
        """Starts the monitoring process."""

        self.monitor.start()
        return self

    def __exit__(self, *exc):
        """Stops the monitoring process and returns the summary of
        observations."""

        self.event.set()
        self.monitor.join()
        if self.monitor.exitcode != 0:
            logger.error(
                f"CPU/GPU resource monitor process exited with code "
                f"{self.monitor.exitcode}.  Check logs for errors!"
            )

        try:
            data = self.q.get(timeout=2 * self.interval)
        except queue.Empty:
            logger.warn(
                f"CPU/GPU resource monitor did not provide anything when "
                f"joined (even after a {2*self.interval}-second timeout - "
                f"this is normally due to exceptions on the monitoring process. "
                f"Check above for other exceptions."
            )
            self.data = None
        else:
            # summarize the returned data by creating means
            summary = []
            for k, values in data:
                if values:
                    if k in ("cpu_processes", "cpu_open_files"):
                        summary.append((k, numpy.max(values)))
                    else:
                        summary.append((k, numpy.mean(values)))
                else:
                    summary.append((k, 0.0))
            self.data = tuple(summary)
