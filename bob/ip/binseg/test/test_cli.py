#!/usr/bin/env python
# coding=utf-8

"""Tests for our CLI applications"""

import re
import contextlib

from click.testing import CliRunner

## special trick for CI builds
from . import mock_dataset
_, rc_variable_set = mock_dataset()


@contextlib.contextmanager
def stdout_logging():

    ## copy logging messages to std out
    import sys
    import logging
    import io
    buf = io.StringIO()
    ch = logging.StreamHandler(buf)
    ch.setFormatter(logging.Formatter('%(message)s'))
    ch.setLevel(logging.INFO)
    logger = logging.getLogger('bob')
    logger.addHandler(ch)
    yield buf
    logger.removeHandler(ch)


def _assert_exit_0(result):

    assert result.exit_code == 0, (
            f"Exit code != 0 ({result.exit_code}); Output:\n{result.output}"
            )

def _check_help(entry_point):

    runner = CliRunner()
    result = runner.invoke(entry_point, ["--help"])
    _assert_exit_0(result)
    assert result.output.startswith("Usage:")


def test_main_help():
    from ..script.binseg import binseg

    _check_help(binseg)


def test_experiment_help():
    from ..script.experiment import experiment

    _check_help(experiment)


def _str_counter(substr, s):
    return sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(substr), s))


@rc_variable_set("bob.ip.binseg.stare.datadir")
def test_experiment_stare():
    from ..script.experiment import experiment

    runner = CliRunner()
    with runner.isolated_filesystem(), stdout_logging() as buf:
        result = runner.invoke(experiment, ["m2unet", "stare", "-vv",
            "--epochs=1", "--batch-size=1", "--overlayed"])
        _assert_exit_0(result)
        keywords = {  #from different logging systems
            "Started training": 1,  #logging
            "epoch: 1|total-time": 1,  #logging
            "Saving checkpoint to results/model/model_final.pth": 1,  #logging
            "Ended training": 1,  #logging
            "Started prediction": 1,  #logging
            "Loading checkpoint from": 2,  #logging
            #"Saving results/overlayed/probabilities": 1,  #tqdm.write
            "Ended prediction": 1,  #logging
            "Started evaluation": 1,  #logging
            "Highest F1-score of": 2,  #logging
            "Saving overall precision-recall plot": 2,  #logging
            #"Saving results/overlayed/analysis": 1,  #tqdm.write
            "Ended evaluation": 1,  #logging
            "Started comparison": 1,  #logging
            "Loading metrics from results/analysis": 2,  #logging
            "Ended comparison": 1,  #logging
            }
        buf.seek(0)
        logging_output = buf.read()
        for k,v in keywords.items():
            #if _str_counter(k, logging_output) != v:
            #    print(f"Count for string '{k}' appeared " \
            #        f"({_str_counter(k, result.output)}) " \
            #        f"instead of the expected {v}")
            assert _str_counter(k, logging_output) == v, \
                    f"Count for string '{k}' appeared " \
                    f"({_str_counter(k, result.output)}) " \
                    f"instead of the expected {v}"


def test_train_help():
    from ..script.train import train

    _check_help(train)


def test_predict_help():
    from ..script.predict import predict

    _check_help(predict)


def test_evaluate_help():
    from ..script.evaluate import evaluate

    _check_help(evaluate)


def test_compare_help():
    from ..script.compare import compare

    _check_help(compare)


def test_config_help():
    from ..script.config import config

    _check_help(config)


def test_config_list_help():
    from ..script.config import list

    _check_help(list)


def test_config_list():
    from ..script.config import list

    runner = CliRunner()
    result = runner.invoke(list)
    _assert_exit_0(result)
    assert "module: bob.ip.binseg.configs.datasets" in result.output
    assert "module: bob.ip.binseg.configs.models" in result.output


def test_config_list_v():
    from ..script.config import list

    runner = CliRunner()
    result = runner.invoke(list, ["--verbose"])
    _assert_exit_0(result)
    assert "module: bob.ip.binseg.configs.datasets" in result.output
    assert "module: bob.ip.binseg.configs.models" in result.output


def test_config_describe_help():
    from ..script.config import describe

    _check_help(describe)


def test_config_describe_drive():
    from ..script.config import describe

    runner = CliRunner()
    result = runner.invoke(describe, ["drive"])
    _assert_exit_0(result)
    assert "[DRIVE-2004]" in result.output


def test_config_copy_help():
    from ..script.config import copy

    _check_help(copy)


def test_config_copy():
    from ..script.config import copy

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(copy, ["drive", "test.py"])
        _assert_exit_0(result)
        with open("test.py") as f:
            data = f.read()
        assert "[DRIVE-2004]" in data


def test_dataset_help():
    from ..script.dataset import dataset

    _check_help(dataset)


def test_dataset_list_help():
    from ..script.dataset import list

    _check_help(list)


def test_dataset_list():
    from ..script.dataset import list

    runner = CliRunner()
    result = runner.invoke(list)
    _assert_exit_0(result)
    assert result.output.startswith("Supported datasets:")


def test_dataset_check_help():
    from ..script.dataset import check

    _check_help(check)


def test_dataset_check():
    from ..script.dataset import check

    runner = CliRunner()
    result = runner.invoke(check, ["--verbose", "--verbose", "--limit=2"])
    _assert_exit_0(result)
