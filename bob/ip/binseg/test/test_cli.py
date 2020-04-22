#!/usr/bin/env python
# coding=utf-8

"""Tests for our CLI applications"""

import os
import re
import fnmatch
import tempfile
import contextlib

import nose.tools

from click.testing import CliRunner

from . import mock_dataset

stare_datadir, stare_dataset, rc_variable_set = mock_dataset()


@contextlib.contextmanager
def stdout_logging():

    ## copy logging messages to std out
    import sys
    import logging
    import io

    buf = io.StringIO()
    ch = logging.StreamHandler(buf)
    ch.setFormatter(logging.Formatter("%(message)s"))
    ch.setLevel(logging.INFO)
    logger = logging.getLogger("bob")
    logger.addHandler(ch)
    yield buf
    logger.removeHandler(ch)


def _assert_exit_0(result):

    assert (
        result.exit_code == 0
    ), f"Exit code {result.exit_code} != 0 -- Output:\n{result.output}"


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
    return sum(1 for _ in re.finditer(r"%s" % re.escape(substr), s))


@rc_variable_set("bob.ip.binseg.stare.datadir")
def test_experiment_stare():

    from ..script.experiment import experiment

    runner = CliRunner()
    with runner.isolated_filesystem(), stdout_logging() as buf, tempfile.NamedTemporaryFile(
        mode="wt"
    ) as config:

        # re-write STARE dataset configuration for test
        config.write("from bob.ip.binseg.data.stare import _make_dataset\n")
        config.write(f"_raw = _make_dataset('{stare_datadir}')\n")
        config.write(
            "from bob.ip.binseg.configs.datasets.stare import _maker\n"
        )
        config.write("dataset = _maker('ah', _raw)\n")
        config.write("second_annotator = _maker('vk', _raw)\n")
        config.flush()

        output_folder = "results"
        result = runner.invoke(
            experiment,
            [
                "m2unet",
                config.name,
                "-vv",
                "--epochs=1",
                "--batch-size=1",
                "--overlayed",
                f"--output-folder={output_folder}",
            ],
        )
        _assert_exit_0(result)

        # check command-line
        assert os.path.exists(os.path.join(output_folder, "command.sh"))

        # check model was saved
        train_folder = os.path.join(output_folder, "model")
        assert os.path.exists(os.path.join(train_folder, "model_final.pth"))
        assert os.path.exists(os.path.join(train_folder, "last_checkpoint"))
        assert os.path.exists(os.path.join(train_folder, "trainlog.csv"))

        # check predictions are there
        predict_folder = os.path.join(output_folder, "predictions")
        assert os.path.exists(os.path.join(predict_folder, "model-info.txt"))
        basedir = os.path.join(predict_folder, "stare-images")
        assert os.path.exists(basedir)
        nose.tools.eq_(len(fnmatch.filter(os.listdir(basedir), "*.hdf5")), 20)

        # check overlayed images are there (since we requested them)
        overlay_folder = os.path.join(output_folder, "overlayed", "predictions")
        basedir = os.path.join(overlay_folder, "stare-images")
        assert os.path.exists(basedir)
        nose.tools.eq_(len(fnmatch.filter(os.listdir(basedir), "*.png")), 20)

        # check evaluation outputs
        eval_folder = os.path.join(output_folder, "analysis")
        second_folder = os.path.join(eval_folder, "second-annotator")
        assert os.path.exists(os.path.join(eval_folder, "train", "metrics.csv"))
        assert os.path.exists(os.path.join(eval_folder, "test", "metrics.csv"))
        assert os.path.exists(os.path.join(second_folder, "train", "metrics.csv"))
        assert os.path.exists(os.path.join(second_folder, "test", "metrics.csv"))

        # check overlayed images are there (since we requested them)
        overlay_folder = os.path.join(output_folder, "overlayed", "analysis")
        basedir = os.path.join(overlay_folder, "stare-images")
        assert os.path.exists(basedir)
        nose.tools.eq_(len(fnmatch.filter(os.listdir(basedir), "*.png")), 20)

        # check overlayed images from first-to-second annotator comparisons are
        # there (since we requested them)
        overlay_folder = os.path.join(output_folder, "overlayed", "analysis",
                "second-annotator")
        basedir = os.path.join(overlay_folder, "stare-images")
        assert os.path.exists(basedir)
        nose.tools.eq_(len(fnmatch.filter(os.listdir(basedir), "*.png")), 20)

        # check outcomes of the comparison phase
        assert os.path.exists(os.path.join(output_folder, "comparison.pdf"))

        keywords = {  # from different logging systems
            "Started training": 1,  # logging
            "Found (dedicated) '__train__' set for training": 1,  # logging
            "epoch: 1|total-time": 1,  # logging
            "Saving checkpoint": 1,  # logging
            "Ended training": 1,  # logging
            "Started prediction": 1,  # logging
            "Loading checkpoint from": 2,  # logging
            # "Saving results/overlayed/probabilities": 1,  #tqdm.write
            "Ended prediction": 1,  # logging
            "Started evaluation": 1,  # logging
            "Highest F1-score of": 4,  # logging
            "Saving overall precision-recall plot": 2,  # logging
            # "Saving results/overlayed/analysis": 1,  #tqdm.write
            "Ended evaluation": 1,  # logging
            "Started comparison": 1,  # logging
            "Loading metrics from": 4,  # logging
            "Ended comparison": 1,  # logging
        }
        buf.seek(0)
        logging_output = buf.read()
        for k, v in keywords.items():
            # if _str_counter(k, logging_output) != v:
            #    print(f"Count for string '{k}' appeared " \
            #        f"({_str_counter(k, result.output)}) " \
            #        f"instead of the expected {v}")
            assert _str_counter(k, logging_output) == v, (
                f"Count for string '{k}' appeared "
                f"({_str_counter(k, logging_output)}) "
                f"instead of the expected {v}"
            )


def _check_train(runner):

    from ..script.train import train

    with tempfile.NamedTemporaryFile(
        mode="wt"
    ) as config, stdout_logging() as buf:

        # single training set configuration
        config.write("from bob.ip.binseg.data.stare import _make_dataset\n")
        config.write(f"_raw = _make_dataset('{stare_datadir}')\n")
        config.write(
            "from bob.ip.binseg.configs.datasets.stare import _maker\n"
        )
        config.write("dataset = _maker('ah', _raw)['train']\n")
        config.flush()

        output_folder = "results"
        result = runner.invoke(
            train,
            ["m2unet", config.name, "-vv", "--epochs=1", "--batch-size=1",
                f"--output-folder={output_folder}"],
        )
        _assert_exit_0(result)

        assert os.path.exists(os.path.join(output_folder, "model_final.pth"))
        assert os.path.exists(os.path.join(output_folder, "last_checkpoint"))
        assert os.path.exists(os.path.join(output_folder, "trainlog.csv"))

        keywords = {  # from different logging systems
            "Continuing from epoch 0": 1,  # logging
            "epoch: 1|total-time": 1,  # logging
            f"Saving checkpoint to {output_folder}/model_final.pth": 1,  # logging
            "Total training time:": 1,  # logging
        }
        buf.seek(0)
        logging_output = buf.read()

        for k, v in keywords.items():
            # if _str_counter(k, logging_output) != v:
            #    print(f"Count for string '{k}' appeared " \
            #        f"({_str_counter(k, result.output)}) " \
            #        f"instead of the expected {v}")
            assert _str_counter(k, logging_output) == v, (
                f"Count for string '{k}' appeared "
                f"({_str_counter(k, logging_output)}) "
                f"instead of the expected {v}:\nOutput:\n{logging_output}"
            )


def _check_predict(runner):

    from ..script.predict import predict

    with tempfile.NamedTemporaryFile(
        mode="wt"
    ) as config, stdout_logging() as buf:

        # single training set configuration
        config.write("from bob.ip.binseg.data.stare import _make_dataset\n")
        config.write(f"_raw = _make_dataset('{stare_datadir}')\n")
        config.write(
            "from bob.ip.binseg.configs.datasets.stare import _maker\n"
        )
        config.write("dataset = _maker('ah', _raw)['test']\n")
        config.flush()

        output_folder = "predictions"
        overlay_folder = os.path.join("overlayed", "predictions")
        result = runner.invoke(
            predict,
            [
                "m2unet",
                config.name,
                "-vv",
                "--batch-size=1",
                "--weight=results/model_final.pth",
                f"--output-folder={output_folder}",
                f"--overlayed={overlay_folder}",
            ],
        )
        _assert_exit_0(result)

        # check predictions are there
        assert os.path.exists(os.path.join(output_folder, "model-info.txt"))
        basedir = os.path.join(output_folder, "stare-images")
        assert os.path.exists(basedir)
        nose.tools.eq_(len(fnmatch.filter(os.listdir(basedir), "*.hdf5")), 10)

        # check overlayed images are there (since we requested them)
        basedir = os.path.join(overlay_folder, "stare-images")
        assert os.path.exists(basedir)
        nose.tools.eq_(len(fnmatch.filter(os.listdir(basedir), "*.png")), 10)

        keywords = {  # from different logging systems
            "Loading checkpoint from": 1,  # logging
            "Total time:": 1,  # logging
        }
        buf.seek(0)
        logging_output = buf.read()

        for k, v in keywords.items():
            # if _str_counter(k, logging_output) != v:
            #    print(f"Count for string '{k}' appeared " \
            #        f"({_str_counter(k, result.output)}) " \
            #        f"instead of the expected {v}")
            assert _str_counter(k, logging_output) == v, (
                f"Count for string '{k}' appeared "
                f"({_str_counter(k, logging_output)}) "
                f"instead of the expected {v}:\nOutput:\n{logging_output}"
            )


def _check_evaluate(runner):

    from ..script.evaluate import evaluate

    with tempfile.NamedTemporaryFile(
        mode="wt"
    ) as config, stdout_logging() as buf:

        # single training set configuration
        config.write("from bob.ip.binseg.data.stare import _make_dataset\n")
        config.write(f"_raw = _make_dataset('{stare_datadir}')\n")
        config.write(
            "from bob.ip.binseg.configs.datasets.stare import _maker\n"
        )
        config.write("dataset = _maker('ah', _raw)['test']\n")
        config.write("second_annotator = _maker('vk', _raw)['test']\n")
        config.flush()

        output_folder = "evaluations"
        second_folder = "evaluations-2nd"
        overlay_folder = os.path.join("overlayed", "analysis")
        result = runner.invoke(
            evaluate,
            [
                config.name,
                "-vv",
                f"--output-folder={output_folder}",
                "--predictions-folder=predictions",
                f"--overlayed={overlay_folder}",
                f"--second-annotator-folder={second_folder}",
            ],
        )
        _assert_exit_0(result)

        assert os.path.exists(os.path.join(output_folder, "metrics.csv"))
        assert os.path.exists(os.path.join(second_folder, "metrics.csv"))

        # check overlayed images are there (since we requested them)
        basedir = os.path.join(overlay_folder, "stare-images")
        assert os.path.exists(basedir)
        nose.tools.eq_(len(fnmatch.filter(os.listdir(basedir), "*.png")), 10)

        keywords = {  # from different logging systems
            "Skipping dataset '__train__'": 0,  # logging
            "Saving averages over all input images": 2,  # logging
            "Highest F1-score": 2,  # logging
        }
        buf.seek(0)
        logging_output = buf.read()

        for k, v in keywords.items():
            # if _str_counter(k, logging_output) != v:
            #    print(f"Count for string '{k}' appeared " \
            #        f"({_str_counter(k, result.output)}) " \
            #        f"instead of the expected {v}")
            assert _str_counter(k, logging_output) == v, (
                f"Count for string '{k}' appeared "
                f"({_str_counter(k, logging_output)}) "
                f"instead of the expected {v}:\nOutput:\n{logging_output}"
            )


def _check_compare(runner):

    from ..script.compare import compare

    with stdout_logging() as buf:

        output_folder = "evaluations"
        second_folder = "evaluations-2nd"
        result = runner.invoke(
            compare,
            [
                "-vv",
                # label - path to metrics
                "test", os.path.join(output_folder, "metrics.csv"),
                "test (2nd. human)", os.path.join(second_folder, "metrics.csv"),
            ],
        )
        _assert_exit_0(result)

        assert os.path.exists("comparison.pdf")

        keywords = {  # from different logging systems
            "Loading metrics from": 2,  # logging
        }
        buf.seek(0)
        logging_output = buf.read()

        for k, v in keywords.items():
            # if _str_counter(k, logging_output) != v:
            #    print(f"Count for string '{k}' appeared " \
            #        f"({_str_counter(k, result.output)}) " \
            #        f"instead of the expected {v}")
            assert _str_counter(k, logging_output) == v, (
                f"Count for string '{k}' appeared "
                f"({_str_counter(k, logging_output)}) "
                f"instead of the expected {v}:\nOutput:\n{logging_output}"
            )


@rc_variable_set("bob.ip.binseg.stare.datadir")
def test_discrete_experiment_stare():

    runner = CliRunner()
    with runner.isolated_filesystem():
        _check_train(runner)
        _check_predict(runner)
        _check_evaluate(runner)
        _check_compare(runner)


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
