#!/usr/bin/env python
# coding=utf-8

"""Tests for our CLI applications"""

from click.testing import CliRunner


def _check_help(entry_point):

    runner = CliRunner()
    result = runner.invoke(entry_point, ["--help"])
    assert result.exit_code == 0
    assert result.output.startswith("Usage:")


def test_main_help():
    from ..script.binseg import binseg

    _check_help(binseg)


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
    assert result.exit_code == 0
    assert "module: bob.ip.binseg.configs.datasets" in result.output
    assert "module: bob.ip.binseg.configs.models" in result.output


def test_config_list_v():
    from ..script.config import list

    runner = CliRunner()
    result = runner.invoke(list, ["--verbose"])
    assert result.exit_code == 0
    assert "module: bob.ip.binseg.configs.datasets" in result.output
    assert "module: bob.ip.binseg.configs.models" in result.output


def test_config_describe_help():
    from ..script.config import describe

    _check_help(describe)


def test_config_describe_drive():
    from ..script.config import describe

    runner = CliRunner()
    result = runner.invoke(describe, ["drive"])
    assert result.exit_code == 0
    assert "[DRIVE-2004]" in result.output


def test_config_copy_help():
    from ..script.config import copy

    _check_help(copy)


def test_config_copy():
    from ..script.config import copy

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(copy, ["drive", "test.py"])
        assert result.exit_code == 0
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
    assert result.exit_code == 0
    assert result.output.startswith("Supported datasets:")


def test_dataset_check_help():
    from ..script.dataset import check

    _check_help(check)


def test_dataset_check():
    from ..script.dataset import check

    runner = CliRunner()
    result = runner.invoke(check, ["--verbose", "--verbose"])
    assert result.exit_code == 0
