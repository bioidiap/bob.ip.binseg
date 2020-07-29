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
    return sum(1 for _ in re.finditer(substr, s, re.MULTILINE))


def _check_experiment_stare(overlay):

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
        options = [
            "m2unet",
            config.name,
            "-vv",
            "--epochs=1",
            "--batch-size=1",
            "--steps=10",
            f"--output-folder={output_folder}",
        ]
        if overlay:
            options += ["--overlayed"]
        result = runner.invoke(experiment, options)
        _assert_exit_0(result)

        # check command-line
        assert os.path.exists(os.path.join(output_folder, "command.sh"))

        # check model was saved
        train_folder = os.path.join(output_folder, "model")
        assert os.path.exists(os.path.join(train_folder, "model_final.pth"))
        assert os.path.exists(
            os.path.join(train_folder, "model_lowest_valid_loss.pth")
        )
        assert os.path.exists(os.path.join(train_folder, "last_checkpoint"))
        assert os.path.exists(os.path.join(train_folder, "constants.csv"))
        assert os.path.exists(os.path.join(train_folder, "trainlog.csv"))
        assert os.path.exists(os.path.join(train_folder, "model_summary.txt"))

        # check predictions are there
        predict_folder = os.path.join(output_folder, "predictions")
        traindir = os.path.join(predict_folder, "train", "stare-images")
        assert os.path.exists(traindir)
        nose.tools.eq_(len(fnmatch.filter(os.listdir(traindir), "*.hdf5")), 10)
        testdir = os.path.join(predict_folder, "test", "stare-images")
        assert os.path.exists(testdir)
        nose.tools.eq_(len(fnmatch.filter(os.listdir(testdir), "*.hdf5")), 10)

        overlay_folder = os.path.join(output_folder, "overlayed", "predictions")
        traindir = os.path.join(overlay_folder, "train", "stare-images")
        testdir = os.path.join(overlay_folder, "test", "stare-images")
        if overlay:
            # check overlayed images are there (since we requested them)
            assert os.path.exists(traindir)
            nose.tools.eq_(
                len(fnmatch.filter(os.listdir(traindir), "*.png")), 10
            )
            # check overlayed images are there (since we requested them)
            assert os.path.exists(testdir)
            nose.tools.eq_(
                len(fnmatch.filter(os.listdir(testdir), "*.png")), 10
            )
        else:
            assert not os.path.exists(traindir)
            assert not os.path.exists(testdir)

        # check evaluation outputs
        eval_folder = os.path.join(output_folder, "analysis")
        assert os.path.exists(os.path.join(eval_folder, "train.csv"))
        # checks individual performance figures are there
        traindir = os.path.join(eval_folder, "train", "stare-images")
        assert os.path.exists(traindir)
        nose.tools.eq_(
            len(fnmatch.filter(os.listdir(traindir), "*.csv")), 10
        )

        assert os.path.exists(os.path.join(eval_folder, "test.csv"))
        # checks individual performance figures are there
        testdir = os.path.join(eval_folder, "test", "stare-images")
        assert os.path.exists(testdir)
        nose.tools.eq_(
            len(fnmatch.filter(os.listdir(testdir), "*.csv")), 10
        )

        assert os.path.exists(
            os.path.join(eval_folder, "second-annotator", "train.csv")
        )
        # checks individual performance figures are there
        traindir_sa = os.path.join(eval_folder, "second-annotator", "train",
                "stare-images")
        assert os.path.exists(traindir_sa)
        nose.tools.eq_(
            len(fnmatch.filter(os.listdir(traindir_sa), "*.csv")), 10
        )

        assert os.path.exists(
            os.path.join(eval_folder, "second-annotator", "test.csv")
        )
        testdir_sa = os.path.join(eval_folder, "second-annotator", "test",
                "stare-images")
        assert os.path.exists(testdir_sa)
        nose.tools.eq_(
            len(fnmatch.filter(os.listdir(testdir_sa), "*.csv")), 10
        )

        overlay_folder = os.path.join(output_folder, "overlayed", "analysis")
        traindir = os.path.join(overlay_folder, "train", "stare-images")
        testdir = os.path.join(overlay_folder, "test", "stare-images")
        if overlay:
            # check overlayed images are there (since we requested them)
            assert os.path.exists(traindir)
            nose.tools.eq_(
                len(fnmatch.filter(os.listdir(traindir), "*.png")), 10
            )
            assert os.path.exists(testdir)
            nose.tools.eq_(
                len(fnmatch.filter(os.listdir(testdir), "*.png")), 10
            )
        else:
            assert not os.path.exists(traindir)
            assert not os.path.exists(testdir)

        # check overlayed images from first-to-second annotator comparisons
        # are there (since we requested them)
        overlay_folder = os.path.join(
            output_folder, "overlayed", "analysis", "second-annotator"
        )
        traindir = os.path.join(overlay_folder, "train", "stare-images")
        testdir = os.path.join(overlay_folder, "test", "stare-images")
        if overlay:
            assert os.path.exists(traindir)
            nose.tools.eq_(
                len(fnmatch.filter(os.listdir(traindir), "*.png")), 10
            )
            assert os.path.exists(testdir)
            nose.tools.eq_(
                len(fnmatch.filter(os.listdir(testdir), "*.png")), 10
            )
        else:
            assert not os.path.exists(traindir)
            assert not os.path.exists(testdir)

        # check outcomes of the comparison phase
        assert os.path.exists(os.path.join(output_folder, "comparison.pdf"))
        assert os.path.exists(os.path.join(output_folder, "comparison.rst"))

        keywords = {
            r"^Started training$": 1,
            r"^Found \(dedicated\) '__train__' set for training$": 1,
            r"^Found \(dedicated\) '__valid__' set for validation$": 1,
            r"^Will checkpoint lowest loss model on validation set$": 1,
            r"^Continuing from epoch 0$": 1,
            r"^Saving model summary at.*$": 1,
            r"^Model has.*$": 1,
            r"^Found new low on validation set.*$": 1,
            r"^Saving checkpoint": 2,
            r"^Ended training$": 1,
            r"^Started prediction$": 1,
            r"^Loading checkpoint from": 1,
            r"^Ended prediction$": 1,
            r"^Started evaluation$": 1,
            r"^Maximum F1-score of.*\(chosen \*a posteriori\*\)$": 3,
            r"^F1-score of.*\(chosen \*a priori\*\)$": 2,
            r"^F1-score of.*\(second annotator; threshold=0.5\)$": 2,
            r"^Ended evaluation$": 1,
            r"^Started comparison$": 1,
            r"^Loading measures from": 4,
            r"^Creating and saving plot at": 1,
            r"^Tabulating performance summary...": 1,
            r"^Saving table at": 1,
            r"^Ended comparison.*$": 1,
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


@rc_variable_set("bob.ip.binseg.stare.datadir")
def test_experiment_stare_with_overlay():
    _check_experiment_stare(overlay=True)


@rc_variable_set("bob.ip.binseg.stare.datadir")
def test_experiment_stare_without_overlay():
    _check_experiment_stare(overlay=False)


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
        config.write("dataset = _maker('ah', _raw)\n")
        config.flush()

        output_folder = "results"
        result = runner.invoke(
            train,
            [
                "m2unet",
                config.name,
                "-vv",
                "--epochs=1",
                "--batch-size=1",
                f"--output-folder={output_folder}",
            ],
        )
        _assert_exit_0(result)

        assert os.path.exists(os.path.join(output_folder, "model_final.pth"))
        assert os.path.exists(
            os.path.join(output_folder, "model_lowest_valid_loss.pth")
        )
        assert os.path.exists(os.path.join(output_folder, "last_checkpoint"))
        assert os.path.exists(os.path.join(output_folder, "constants.csv"))
        assert os.path.exists(os.path.join(output_folder, "trainlog.csv"))
        assert os.path.exists(os.path.join(output_folder, "model_summary.txt"))

        keywords = {
            r"^Found \(dedicated\) '__train__' set for training$": 1,
            r"^Found \(dedicated\) '__valid__' set for validation$": 1,
            r"^Continuing from epoch 0$": 1,
            r"^Saving model summary at.*$": 1,
            r"^Model has.*$": 1,
            r"^Saving checkpoint to .*/model_lowest_valid_loss.pth$": 1,
            r"^Saving checkpoint to .*/model_final.pth$": 1,
            r"^Total training time:": 1,
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
        basedir = os.path.join(output_folder, "test", "stare-images")
        assert os.path.exists(basedir)
        nose.tools.eq_(len(fnmatch.filter(os.listdir(basedir), "*.hdf5")), 10)

        # check overlayed images are there (since we requested them)
        basedir = os.path.join(overlay_folder, "test", "stare-images")
        assert os.path.exists(basedir)
        nose.tools.eq_(len(fnmatch.filter(os.listdir(basedir), "*.png")), 10)

        keywords = {
            r"^Loading checkpoint from.*$": 1,
            r"^Total time:.*$": 1,
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
        overlay_folder = os.path.join("overlayed", "analysis")
        result = runner.invoke(
            evaluate,
            [
                config.name,
                "-vv",
                "--steps=10",
                f"--output-folder={output_folder}",
                "--predictions-folder=predictions",
                f"--overlayed={overlay_folder}",
            ],
        )
        _assert_exit_0(result)

        assert os.path.exists(os.path.join(output_folder, "test.csv"))
        # checks individual performance figures are there
        testdir = os.path.join(output_folder, "test", "stare-images")
        assert os.path.exists(testdir)
        nose.tools.eq_(
            len(fnmatch.filter(os.listdir(testdir), "*.csv")), 10
        )

        assert os.path.exists(
            os.path.join(output_folder, "second-annotator", "test.csv")
        )
        # checks individual performance figures are there
        testdir_sa = os.path.join(output_folder, "second-annotator", "test",
                "stare-images")
        assert os.path.exists(testdir_sa)
        nose.tools.eq_(
            len(fnmatch.filter(os.listdir(testdir_sa), "*.csv")), 10
        )

        # check overlayed images are there (since we requested them)
        basedir = os.path.join(overlay_folder, "test", "stare-images")
        assert os.path.exists(basedir)
        nose.tools.eq_(len(fnmatch.filter(os.listdir(basedir), "*.png")), 10)

        keywords = {
            r"^Skipping dataset '__train__'": 0,
            r"^Saving averages over all input images.*$": 2,
            r"^Maximum F1-score of.*\(chosen \*a posteriori\*\)$": 1,
            r"^F1-score of.*\(chosen \*a priori\*\)$": 1,
            r"^F1-score of.*\(second annotator; threshold=0.5\)$": 1,
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
        result = runner.invoke(
            compare,
            [
                "-vv",
                # label - path to measures
                "test",
                os.path.join(output_folder, "test.csv"),
                "test (2nd. human)",
                os.path.join(output_folder, "second-annotator", "test.csv"),
                "--output-figure=comparison.pdf",
                "--output-table=comparison.rst",
            ],
        )
        _assert_exit_0(result)

        assert os.path.exists("comparison.pdf")
        assert os.path.exists("comparison.rst")

        keywords = {
            r"^Loading measures from": 2,
            r"^Creating and saving plot at": 1,
            r"^Tabulating performance summary...": 1,
            r"^Saving table at": 1,
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


def _check_significance(runner):

    from ..script.significance import significance

    with tempfile.NamedTemporaryFile(
        mode="wt"
    ) as config, stdout_logging() as buf:

        config.write("from bob.ip.binseg.data.stare import _make_dataset\n")
        config.write(f"_raw = _make_dataset('{stare_datadir}')\n")
        config.write(
            "from bob.ip.binseg.configs.datasets.stare import _maker\n"
        )
        config.write("dataset = _maker('ah', _raw)\n")
        config.flush()

        ofolder = "significance"
        cfolder = os.path.join(ofolder, "caches")

        result = runner.invoke(
            significance,
            [
                "-vv",
                config.name,
                "--names=v1", "v2",
                "--predictions=predictions", "predictions",
                "--threshold=0.5",
                "--size=64", "64",
                "--stride=32", "32",
                "--figure=accuracy",
                f"--output-folder={ofolder}",
                f"--checkpoint-folder={cfolder}",
            ],
        )
        _assert_exit_0(result)

        assert os.path.exists(ofolder)
        assert os.path.exists(cfolder)
        assert os.path.exists(os.path.join(ofolder, "analysis.pdf"))
        assert os.path.exists(os.path.join(ofolder, "analysis.txt"))

        keywords = {
            r"^Evaluating sliding window 'accuracy' on": 2,
            r"^Evaluating sliding window 'accuracy' differences on": 1,
            #r"^Basic statistics from distributions:$": 1,
            r"^Writing analysis figures": 1,
            r"^Writing analysis summary": 1,
            r"^Differences are exactly zero": 2,
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
        _check_significance(runner)


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


def test_significance_help():
    from ..script.significance import significance

    _check_help(significance)


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
