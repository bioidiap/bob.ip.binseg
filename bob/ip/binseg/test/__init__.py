#!/usr/bin/env python
# coding=utf-8

"""Unit tests"""

import tempfile
import logging

logger = logging.getLogger(__name__)

TESTDB_TMPDIR = None
_URL = (
    "http://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/_testdb.zip"
)
_RCKEY = "bob.ip.binseg.stare.datadir"


def teardown_package():
    global TESTDB_TMPDIR
    if TESTDB_TMPDIR is not None:
        logger.info(f"Removing temporary directory {TESTDB_TMPDIR.name}...")
        TESTDB_TMPDIR.cleanup()


def mock_dataset():
    global TESTDB_TMPDIR
    from bob.extension import rc

    if (TESTDB_TMPDIR is not None) or (_RCKEY in rc):
        logger.info("Test database already set up - not downloading")
    else:
        logger.info("Test database not available, downloading...")
        import zipfile
        import urllib.request

        # Download the file from `url` and save it locally under `file_name`:
        with urllib.request.urlopen(_URL) as r, tempfile.TemporaryFile() as f:
            f.write(r.read())
            f.flush()
            f.seek(0)
            TESTDB_TMPDIR = tempfile.TemporaryDirectory(prefix=__name__ + "-")
            print(f"Creating test database at {TESTDB_TMPDIR.name}...")
            logger.info(f"Creating test database at {TESTDB_TMPDIR.name}...")
            with zipfile.ZipFile(f) as zf:
                zf.extractall(TESTDB_TMPDIR.name)

    from ..data import stare

    if TESTDB_TMPDIR is None:
        # if the user has the STARE directory ready, then we do a normal return
        return rc["bob.ip.binseg.stare.datadir"], stare.dataset

    # else, we do a "mock" return
    return (
        TESTDB_TMPDIR.name,
        stare._make_dataset(TESTDB_TMPDIR.name),
    )
