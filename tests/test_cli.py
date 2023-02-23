#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
# SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
# SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
# SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
# SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for our CLI applications."""

import fnmatch
import logging
import os
import re
import tempfile

from click.testing import CliRunner

from tests.utils import assert_click_runner_result


def _assert_exit_0(result):
    assert_click_runner_result(result)


def _check_help(entry_point):
    runner = CliRunner()
    result = runner.invoke(entry_point, ["--help"])
    _assert_exit_0(result)
    assert result.output.startswith("Usage:")


def test_main_help_binseg():
    from deepdraw.common.script.common import binseg

    _check_help(binseg)


def test_main_help_detect():
    from deepdraw.common.script.common import detect

    _check_help(detect)


def test_binseg_experiment_help():
    from deepdraw.binseg.script.experiment import experiment

    _check_help(experiment)


def test_detect_experiment_help():
    from deepdraw.detect.script.experiment import experiment

    _check_help(experiment)


def _str_counter(substr, s):
    return sum(1 for _ in re.finditer(substr, s, re.MULTILINE))


def _check_experiment_stare(
    caplog, stare_datadir, overlay, multiprocess=False, extra_valid=0
):
    from deepdraw.binseg.script.experiment import experiment

    # ensures we capture only ERROR messages and above by default
    caplog.set_level(logging.ERROR)

    runner = CliRunner()
    with runner.isolated_filesystem(), caplog.at_level(
        logging.INFO, logger="deepdraw.binseg"
    ), tempfile.NamedTemporaryFile(mode="wt") as config:
        # re-write STARE dataset configuration for test
        config.write("from deepdraw.binseg.data.stare import _make_dataset\n")
        config.write(f"_raw = _make_dataset('{stare_datadir}')\n")
        config.write(
            "from deepdraw.binseg.configs.datasets.stare import _maker\n"
        )
        config.write("dataset = _maker('ah', _raw)\n")
        if extra_valid > 0:
            # simulates the existence of a single extra validation dataset
            # which is simply a copy of the __valid__ dataset for this test...
            config.write(
                f"dataset['__extra_valid__'] = "
                f"{extra_valid}*[dataset['__valid__']]\n"
            )
        config.write("second_annotator = _maker('vk', _raw)\n")
        config.flush()

        output_folder = "results"
        options = [
            "lwnet",
            config.name,
            "-vv",
            "--epochs=1",
            "--batch-size=1",
            "--steps=10",
            f"--output-folder={output_folder}",
            "--monitoring-interval=2",
            "--plot-limits=0.1",
            "1.0",
            "0.1",
            "1.0",
        ]
        if overlay:
            options += ["--overlayed"]
        if multiprocess:
            options += ["--parallel=1"]

        result = runner.invoke(experiment, options)

        _assert_exit_0(result)

        # check command-line
        assert os.path.exists(os.path.join(output_folder, "command.sh"))

        # check model was saved
        train_folder = os.path.join(output_folder, "model")
        assert os.path.exists(
            os.path.join(train_folder, "model_final_epoch.pth")
        )
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
        assert len(fnmatch.filter(os.listdir(traindir), "*.hdf5")) == 10
        testdir = os.path.join(predict_folder, "test", "stare-images")
        assert os.path.exists(testdir)
        assert len(fnmatch.filter(os.listdir(testdir), "*.hdf5")) == 10

        overlay_folder = os.path.join(output_folder, "overlayed", "predictions")
        traindir = os.path.join(overlay_folder, "train", "stare-images")
        testdir = os.path.join(overlay_folder, "test", "stare-images")
        if overlay:
            # check overlayed images are there (since we requested them)
            assert os.path.exists(traindir)
            assert len(fnmatch.filter(os.listdir(traindir), "*.png")) == 10
            # check overlayed images are there (since we requested them)
            assert os.path.exists(testdir)
            assert len(fnmatch.filter(os.listdir(testdir), "*.png")) == 10
        else:
            assert not os.path.exists(traindir)
            assert not os.path.exists(testdir)

        # check evaluation outputs
        eval_folder = os.path.join(output_folder, "analysis")
        assert os.path.exists(os.path.join(eval_folder, "train.csv"))
        # checks individual performance figures are there
        traindir = os.path.join(eval_folder, "train", "stare-images")
        assert os.path.exists(traindir)
        assert len(fnmatch.filter(os.listdir(traindir), "*.csv")) == 10

        assert os.path.exists(os.path.join(eval_folder, "test.csv"))
        # checks individual performance figures are there
        testdir = os.path.join(eval_folder, "test", "stare-images")
        assert os.path.exists(testdir)
        assert len(fnmatch.filter(os.listdir(testdir), "*.csv")) == 10

        assert os.path.exists(
            os.path.join(eval_folder, "second-annotator", "train.csv")
        )
        # checks individual performance figures are there
        traindir_sa = os.path.join(
            eval_folder, "second-annotator", "train", "stare-images"
        )
        assert os.path.exists(traindir_sa)
        assert len(fnmatch.filter(os.listdir(traindir_sa), "*.csv")) == 10

        assert os.path.exists(
            os.path.join(eval_folder, "second-annotator", "test.csv")
        )
        testdir_sa = os.path.join(
            eval_folder, "second-annotator", "test", "stare-images"
        )
        assert os.path.exists(testdir_sa)
        assert len(fnmatch.filter(os.listdir(testdir_sa), "*.csv")) == 10

        overlay_folder = os.path.join(output_folder, "overlayed", "analysis")
        traindir = os.path.join(overlay_folder, "train", "stare-images")
        testdir = os.path.join(overlay_folder, "test", "stare-images")
        if overlay:
            # check overlayed images are there (since we requested them)
            assert os.path.exists(traindir)
            assert len(fnmatch.filter(os.listdir(traindir), "*.png")) == 10
            assert os.path.exists(testdir)
            assert len(fnmatch.filter(os.listdir(testdir), "*.png")) == 10
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
            assert len(fnmatch.filter(os.listdir(traindir), "*.png")) == 10
            assert os.path.exists(testdir)
            assert len(fnmatch.filter(os.listdir(testdir), "*.png")) == 10
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
            f"^Found {extra_valid} extra validation": 1 if extra_valid else 0,
            r"^Extra validation sets are NOT used for model checkpointing": 1
            if extra_valid
            else 0,
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
        messages = "\n".join([k.getMessage() for k in caplog.records])
        for k, v in keywords.items():
            total = _str_counter(k, messages)
            assert total == v, (
                f"message '{k}' appears {total} times, but I expected "
                f"it to appear {v} times"
            )


def test_experiment_stare_with_overlay(caplog, stare_datadir):
    _check_experiment_stare(caplog, stare_datadir, overlay=True)


def test_experiment_stare_without_overlay(caplog, stare_datadir):
    _check_experiment_stare(caplog, stare_datadir, overlay=False)


def test_experiment_stare_with_multiprocessing(caplog, stare_datadir):
    _check_experiment_stare(
        caplog, stare_datadir, overlay=False, multiprocess=True
    )


def test_experiment_stare_with_extra_validation(caplog, stare_datadir):
    _check_experiment_stare(caplog, stare_datadir, overlay=False, extra_valid=1)


def test_experiment_stare_with_multiple_extra_validation(caplog, stare_datadir):
    _check_experiment_stare(caplog, stare_datadir, overlay=False, extra_valid=3)


def _check_experiment_stare_detection(
    caplog, stare_datadir, overlay, multiprocess=False, extra_valid=0
):
    from deepdraw.detect.script.experiment import experiment

    # ensures we capture only ERROR messages and above by default
    caplog.set_level(logging.ERROR)

    runner = CliRunner()
    with runner.isolated_filesystem(), caplog.at_level(
        logging.INFO, logger="deepdraw.detect"
    ), tempfile.NamedTemporaryFile(mode="wt") as config:
        # re-write STARE dataset configuration for test
        config.write("from deepdraw.binseg.data.stare import _make_dataset\n")
        config.write(f"_raw = _make_dataset('{stare_datadir}')\n")
        config.write(
            "from deepdraw.detect.configs.datasets.stare import _maker\n"
        )
        config.write("dataset = _maker('ah', _raw)\n")
        if extra_valid > 0:
            # simulates the existence of a single extra validation dataset
            # which is simply a copy of the __valid__ dataset for this test...
            config.write(
                f"dataset['__extra_valid__'] = "
                f"{extra_valid}*[dataset['__valid__']]\n"
            )
        config.flush()

        output_folder = "results"
        options = [
            "faster-rcnn",
            config.name,
            "-vv",
            "--epochs=1",
            "--batch-size=1",
            "--steps=10",
            f"--output-folder={output_folder}",
            "--monitoring-interval=2",
        ]
        if overlay:
            options += ["--overlayed"]
        if multiprocess:
            options += ["--parallel=1"]

        result = runner.invoke(experiment, options)
        _assert_exit_0(result)

        # check command-line
        assert os.path.exists(os.path.join(output_folder, "command.sh"))

        # check model was saved
        train_folder = os.path.join(output_folder, "model")
        assert os.path.exists(
            os.path.join(train_folder, "model_final_epoch.pth")
        )
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
        assert len(fnmatch.filter(os.listdir(traindir), "*.hdf5")) == 10
        testdir = os.path.join(predict_folder, "test", "stare-images")
        assert os.path.exists(testdir)
        assert len(fnmatch.filter(os.listdir(testdir), "*.hdf5")) == 10

        overlay_folder = os.path.join(output_folder, "overlayed", "predictions")
        traindir = os.path.join(overlay_folder, "train", "stare-images")
        testdir = os.path.join(overlay_folder, "test", "stare-images")
        if overlay:
            # check overlayed images are there (since we requested them)
            assert os.path.exists(traindir)
            assert len(fnmatch.filter(os.listdir(traindir), "*.png")) == 10
            # check overlayed images are there (since we requested them)
            assert os.path.exists(testdir)
            assert len(fnmatch.filter(os.listdir(testdir), "*.png")) == 10
        else:
            assert not os.path.exists(traindir)
            assert not os.path.exists(testdir)

        # check evaluation outputs
        eval_folder = os.path.join(output_folder, "analysis")
        assert os.path.exists(os.path.join(eval_folder, "train.csv"))
        # checks individual performance figures are there
        traindir = os.path.join(eval_folder, "train", "stare-images")
        assert os.path.exists(traindir)
        assert len(fnmatch.filter(os.listdir(traindir), "*.csv")) == 10

        assert os.path.exists(os.path.join(eval_folder, "test.csv"))
        # checks individual performance figures are there
        testdir = os.path.join(eval_folder, "test", "stare-images")
        assert os.path.exists(testdir)
        assert len(fnmatch.filter(os.listdir(testdir), "*.csv")) == 10

        overlay_folder = os.path.join(output_folder, "overlayed", "analysis")
        traindir = os.path.join(overlay_folder, "train", "stare-images")
        testdir = os.path.join(overlay_folder, "test", "stare-images")
        if overlay:
            # check overlayed images are there (since we requested them)
            assert os.path.exists(traindir)
            assert len(fnmatch.filter(os.listdir(traindir), "*.png")) == 10
            assert os.path.exists(testdir)
            assert len(fnmatch.filter(os.listdir(testdir), "*.png")) == 10
        else:
            assert not os.path.exists(traindir)
            assert not os.path.exists(testdir)

        # check overlayed images from first-to-second annotator comparisons
        # are there (since we requested them)
        traindir = os.path.join(overlay_folder, "train", "stare-images")
        testdir = os.path.join(overlay_folder, "test", "stare-images")
        if overlay:
            assert os.path.exists(traindir)
            assert len(fnmatch.filter(os.listdir(traindir), "*.png")) == 10
            assert os.path.exists(testdir)
            assert len(fnmatch.filter(os.listdir(testdir), "*.png")) == 10
        else:
            assert not os.path.exists(traindir)
            assert not os.path.exists(testdir)

        keywords = {
            r"^Started training$": 1,
            r"^Found \(dedicated\) '__train__' set for training$": 1,
            r"^Found \(dedicated\) '__valid__' set for validation$": 1,
            r"^Will checkpoint lowest loss model on validation set$": 1,
            f"^Found {extra_valid} extra validation": 1 if extra_valid else 0,
            r"^Extra validation sets are NOT used for model checkpointing": 1
            if extra_valid
            else 0,
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
            r"^Maximum IoU of.*\(chosen \*a posteriori\*\)$": 3,
            r"^IoU of.*\(chosen \*a priori\*\)$": 2,
            r"^Ended evaluation$": 1,
            r"^Started comparison$": 1,
            r"^Loading measures from": 2,
            r"^Tabulating performance summary...": 1,
            r"^Saving table at": 1,
            r"^Ended comparison.*$": 1,
        }
        messages = "\n".join([k.getMessage() for k in caplog.records])
        for k, v in keywords.items():
            total = _str_counter(k, messages)
            assert total == v, (
                f"message '{k}' appears {total} times, but I expected "
                f"it to appear {v} times"
            )


def test_experiment_stare_with_overlay_detection(caplog, stare_datadir):
    _check_experiment_stare_detection(caplog, stare_datadir, overlay=True)


def test_experiment_stare_without_overlay_detection(caplog, stare_datadir):
    _check_experiment_stare_detection(caplog, stare_datadir, overlay=False)


def test_experiment_stare_with_multiprocessing_detection(caplog, stare_datadir):
    _check_experiment_stare_detection(
        caplog, stare_datadir, overlay=False, multiprocess=True
    )


def test_experiment_stare_with_extra_validation_detection(
    caplog, stare_datadir
):
    _check_experiment_stare_detection(
        caplog, stare_datadir, overlay=False, extra_valid=1
    )


def _check_train(caplog, runner, stare_datadir):
    from deepdraw.binseg.script.train import train

    with tempfile.NamedTemporaryFile(mode="wt") as config, caplog.at_level(
        logging.INFO, logger="deepdraw.binseg"
    ):
        # single training set configuration
        config.write("from deepdraw.binseg.data.stare import _make_dataset\n")
        config.write(f"_raw = _make_dataset('{stare_datadir}')\n")
        config.write(
            "from deepdraw.binseg.configs.datasets.stare import _maker\n"
        )
        config.write("dataset = _maker('ah', _raw)\n")
        config.flush()

        output_folder = "results"
        result = runner.invoke(
            train,
            [
                "lwnet",
                config.name,
                "-vv",
                "--epochs=1",
                "--batch-size=1",
                f"--output-folder={output_folder}",
            ],
        )
        _assert_exit_0(result)

        assert os.path.exists(
            os.path.join(output_folder, "model_final_epoch.pth")
        )
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
            r"^Saving checkpoint to .*/model_final_epoch.pth$": 1,
            r"^Total training time:": 1,
        }

        messages = "\n".join([k.getMessage() for k in caplog.records])
        for k, v in keywords.items():
            total = _str_counter(k, messages)
            assert total == v, (
                f"message '{k}' appears {total} times, but I expected "
                f"it to appear {v} times"
            )


def _check_predict(caplog, runner, stare_datadir):
    from deepdraw.binseg.script.predict import predict

    with tempfile.NamedTemporaryFile(mode="wt") as config, caplog.at_level(
        logging.INFO, logger="deepdraw.binseg"
    ):
        # single training set configuration
        config.write("from deepdraw.binseg.data.stare import _make_dataset\n")
        config.write(f"_raw = _make_dataset('{stare_datadir}')\n")
        config.write(
            "from deepdraw.binseg.configs.datasets.stare import _maker\n"
        )
        config.write("dataset = _maker('ah', _raw)['test']\n")
        config.flush()

        output_folder = "predictions"
        overlay_folder = os.path.join("overlayed", "predictions")
        result = runner.invoke(
            predict,
            [
                "lwnet",
                config.name,
                "-vv",
                "--batch-size=1",
                "--weight=results/model_final_epoch.pth",
                f"--output-folder={output_folder}",
                f"--overlayed={overlay_folder}",
            ],
        )
        _assert_exit_0(result)

        # check predictions are there
        basedir = os.path.join(output_folder, "test", "stare-images")
        assert os.path.exists(basedir)
        assert len(fnmatch.filter(os.listdir(basedir), "*.hdf5")) == 10

        # check overlayed images are there (since we requested them)
        basedir = os.path.join(overlay_folder, "test", "stare-images")
        assert os.path.exists(basedir)
        assert len(fnmatch.filter(os.listdir(basedir), "*.png")) == 10

        keywords = {
            r"^Loading checkpoint from.*$": 1,
            r"^Total time:.*$": 1,
        }

        messages = "\n".join([k.getMessage() for k in caplog.records])
        for k, v in keywords.items():
            total = _str_counter(k, messages)
            assert total == v, (
                f"message '{k}' appears {total} times, but I expected "
                f"it to appear {v} times"
            )


def _check_evaluate(caplog, runner, stare_datadir):
    from deepdraw.binseg.script.evaluate import evaluate

    with tempfile.NamedTemporaryFile(mode="wt") as config, caplog.at_level(
        logging.INFO, logger="deepdraw.binseg"
    ):
        # single training set configuration
        config.write("from deepdraw.binseg.data.stare import _make_dataset\n")
        config.write(f"_raw = _make_dataset('{stare_datadir}')\n")
        config.write(
            "from deepdraw.binseg.configs.datasets.stare import _maker\n"
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
        assert len(fnmatch.filter(os.listdir(testdir), "*.csv")) == 10

        assert os.path.exists(
            os.path.join(output_folder, "second-annotator", "test.csv")
        )
        # checks individual performance figures are there
        testdir_sa = os.path.join(
            output_folder, "second-annotator", "test", "stare-images"
        )
        assert os.path.exists(testdir_sa)
        assert len(fnmatch.filter(os.listdir(testdir_sa), "*.csv")) == 10

        # check overlayed images are there (since we requested them)
        basedir = os.path.join(overlay_folder, "test", "stare-images")
        assert os.path.exists(basedir)
        assert len(fnmatch.filter(os.listdir(basedir), "*.png")) == 10

        keywords = {
            r"^Maximum F1-score of.*\(chosen \*a posteriori\*\)$": 1,
            r"^F1-score of.*\(chosen \*a priori\*\)$": 1,
            r"^F1-score of.*\(second annotator; threshold=0.5\)$": 1,
        }

        messages = "\n".join([k.getMessage() for k in caplog.records])
        for k, v in keywords.items():
            total = _str_counter(k, messages)
            assert total == v, (
                f"message '{k}' appears {total} times, but I expected "
                f"it to appear {v} times"
            )


def _check_compare(caplog, runner):
    from deepdraw.binseg.script.compare import compare

    with caplog.at_level(logging.INFO, logger="deepdraw.binseg"):
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
        messages = "\n".join([k.getMessage() for k in caplog.records])
        for k, v in keywords.items():
            total = _str_counter(k, messages)
            assert total == v, (
                f"message '{k}' appears {total} times, but I expected "
                f"it to appear {v} times"
            )


def _check_significance(caplog, stare_datadir, runner):
    from deepdraw.binseg.script.significance import significance

    with tempfile.NamedTemporaryFile(mode="wt") as config, caplog.at_level(
        logging.INFO, logger="deepdraw.binseg"
    ):
        config.write("from deepdraw.binseg.data.stare import _make_dataset\n")
        config.write(f"_raw = _make_dataset('{stare_datadir}')\n")
        config.write(
            "from deepdraw.binseg.configs.datasets.stare import _maker\n"
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
                "--names=v1",
                "v2",
                "--predictions=predictions",
                "predictions",
                "--threshold=0.5",
                "--size=64",
                "64",
                "--stride=32",
                "32",
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
            # r"^Basic statistics from distributions:$": 1,
            r"^Writing analysis figures": 1,
            r"^Writing analysis summary": 1,
            r"^Differences are exactly zero": 2,
        }
        messages = "\n".join([k.getMessage() for k in caplog.records])
        for k, v in keywords.items():
            total = _str_counter(k, messages)
            assert total == v, (
                f"message '{k}' appears {total} times, but I expected "
                f"it to appear {v} times"
            )


def test_discrete_experiment_stare(caplog, stare_datadir):
    # ensures we capture only ERROR messages and above by default
    caplog.set_level(logging.ERROR)

    runner = CliRunner()
    with runner.isolated_filesystem():
        _check_train(caplog, runner, stare_datadir)
        _check_predict(caplog, runner, stare_datadir)
        _check_evaluate(caplog, runner, stare_datadir)
        _check_compare(caplog, runner)
        # _check_significance(caplog, runner)


def _check_train_detection(caplog, runner, stare_datadir):
    from deepdraw.detect.script.train import train

    with tempfile.NamedTemporaryFile(mode="wt") as config, caplog.at_level(
        logging.INFO, logger="deepdraw.detect"
    ):
        # single training set configuration
        config.write("from deepdraw.binseg.data.stare import _make_dataset\n")
        config.write(f"_raw = _make_dataset('{stare_datadir}')\n")
        config.write(
            "from deepdraw.detect.configs.datasets.stare import _maker\n"
        )
        config.write("dataset = _maker('ah', _raw)\n")
        config.flush()

        output_folder = "results"
        result = runner.invoke(
            train,
            [
                "faster-rcnn",
                config.name,
                "-vv",
                "--epochs=1",
                "--batch-size=1",
                f"--output-folder={output_folder}",
            ],
        )
        _assert_exit_0(result)

        assert os.path.exists(
            os.path.join(output_folder, "model_final_epoch.pth")
        )
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
            r"^Saving checkpoint to .*/model_final_epoch.pth$": 1,
            r"^Total training time:": 1,
        }

        messages = "\n".join([k.getMessage() for k in caplog.records])
        for k, v in keywords.items():
            total = _str_counter(k, messages)
            assert total == v, (
                f"message '{k}' appears {total} times, but I expected "
                f"it to appear {v} times"
            )


def _check_predict_detection(caplog, runner, stare_datadir):
    from deepdraw.detect.script.predict import predict

    with tempfile.NamedTemporaryFile(mode="wt") as config, caplog.at_level(
        logging.INFO, logger="deepdraw.detect"
    ):
        # single training set configuration
        config.write("from deepdraw.binseg.data.stare import _make_dataset\n")
        config.write(f"_raw = _make_dataset('{stare_datadir}')\n")
        config.write(
            "from deepdraw.detect.configs.datasets.stare import _maker\n"
        )
        config.write("dataset = _maker('ah', _raw)['test']\n")
        config.flush()

        output_folder = "predictions"
        overlay_folder = os.path.join("overlayed", "predictions")
        result = runner.invoke(
            predict,
            [
                "faster-rcnn",
                config.name,
                "-vv",
                "--batch-size=1",
                "--weight=results/model_final_epoch.pth",
                f"--output-folder={output_folder}",
                f"--overlayed={overlay_folder}",
            ],
        )
        _assert_exit_0(result)

        # check predictions are there
        basedir = os.path.join(output_folder, "test", "stare-images")
        assert os.path.exists(basedir)
        assert len(fnmatch.filter(os.listdir(basedir), "*.hdf5")) == 10

        # check overlayed images are there (since we requested them)
        basedir = os.path.join(overlay_folder, "test", "stare-images")
        assert os.path.exists(basedir)
        assert len(fnmatch.filter(os.listdir(basedir), "*.png")) == 10

        keywords = {
            r"^Loading checkpoint from.*$": 1,
            r"^Total time:.*$": 1,
        }

        messages = "\n".join([k.getMessage() for k in caplog.records])
        for k, v in keywords.items():
            total = _str_counter(k, messages)
            assert total == v, (
                f"message '{k}' appears {total} times, but I expected "
                f"it to appear {v} times"
            )


def _check_evaluate_detection(caplog, runner, stare_datadir):
    from deepdraw.detect.script.evaluate import evaluate

    with tempfile.NamedTemporaryFile(mode="wt") as config, caplog.at_level(
        logging.INFO, logger="deepdraw.detect"
    ):
        # single training set configuration
        config.write("from deepdraw.binseg.data.stare import _make_dataset\n")
        config.write(f"_raw = _make_dataset('{stare_datadir}')\n")
        config.write(
            "from deepdraw.detect.configs.datasets.stare import _maker\n"
        )
        config.write("dataset = _maker('ah', _raw)['test']\n")
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
        assert len(fnmatch.filter(os.listdir(testdir), "*.csv")) == 10

        # check overlayed images are there (since we requested them)
        basedir = os.path.join(overlay_folder, "test", "stare-images")
        assert os.path.exists(basedir)
        assert len(fnmatch.filter(os.listdir(basedir), "*.png")) == 10

        keywords = {
            r"^Maximum IoU of.*\(chosen \*a posteriori\*\)$": 1,
            r"^IoU of.*\(chosen \*a priori\*\)$": 1,
        }

        messages = "\n".join([k.getMessage() for k in caplog.records])
        for k, v in keywords.items():
            total = _str_counter(k, messages)
            assert total == v, (
                f"message '{k}' appears {total} times, but I expected "
                f"it to appear {v} times"
            )


def test_discrete_experiment_stare_detection(caplog, stare_datadir):
    # ensures we capture only ERROR messages and above by default
    caplog.set_level(logging.ERROR)

    runner = CliRunner()
    with runner.isolated_filesystem():
        _check_train_detection(caplog, runner, stare_datadir)
        _check_predict_detection(caplog, runner, stare_datadir)
        _check_evaluate_detection(caplog, runner, stare_datadir)


def test_train_help():
    from deepdraw.binseg.script.train import train

    _check_help(train)


def test_detect_train_help():
    from deepdraw.detect.script.train import train

    _check_help(train)


def test_predict_help():
    from deepdraw.binseg.script.predict import predict

    _check_help(predict)


def test_detect_predict_help():
    from deepdraw.detect.script.predict import predict

    _check_help(predict)


def test_evaluate_help():
    from deepdraw.binseg.script.evaluate import evaluate

    _check_help(evaluate)


def test_detect_evaluate_help():
    from deepdraw.detect.script.evaluate import evaluate

    _check_help(evaluate)


def test_compare_help():
    from deepdraw.binseg.script.compare import compare

    _check_help(compare)


def test_detect_compare_help():
    from deepdraw.detect.script.compare import compare

    _check_help(compare)


def test_mkmask_help():
    from deepdraw.binseg.script.mkmask import mkmask

    _check_help(mkmask)


def test_significance_help():
    from deepdraw.binseg.script.significance import significance

    _check_help(significance)


def test_config_help():
    from deepdraw.binseg.script.config import config

    _check_help(config)


def test_detect_config_help():
    from deepdraw.detect.script.config import config

    _check_help(config)


def test_config_list_help():
    from deepdraw.binseg.script.config import list

    _check_help(list)


def test_detect_config_list_help():
    from deepdraw.detect.script.config import list

    _check_help(list)


def test_config_list():
    from deepdraw.binseg.script.config import list

    runner = CliRunner()
    result = runner.invoke(list)
    _assert_exit_0(result)
    assert "module: deepdraw.binseg.configs.datasets" in result.output
    assert "module: deepdraw.binseg.configs.models" in result.output


def test_detect_config_list():
    from deepdraw.detect.script.config import list

    runner = CliRunner()
    result = runner.invoke(list)
    _assert_exit_0(result)
    assert "module: deepdraw.detect.configs.datasets" in result.output
    assert "module: deepdraw.detect.configs.models" in result.output


def test_config_list_v():
    from deepdraw.binseg.script.config import list

    runner = CliRunner()
    result = runner.invoke(list, ["--verbose"])
    _assert_exit_0(result)
    assert "module: deepdraw.binseg.configs.datasets" in result.output
    assert "module: deepdraw.binseg.configs.models" in result.output


def test_config_describe_help():
    from deepdraw.binseg.script.config import describe

    _check_help(describe)


def test_detect_config_describe_help():
    from deepdraw.detect.script.config import describe

    _check_help(describe)


def test_config_describe_drive():
    from deepdraw.binseg.script.config import describe

    runner = CliRunner()
    result = runner.invoke(describe, ["drive"])
    _assert_exit_0(result)
    assert "[DRIVE-2004]" in result.output


def test_config_copy_help():
    from deepdraw.binseg.script.config import copy

    _check_help(copy)


def test_config_copy():
    from deepdraw.binseg.script.config import copy

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(copy, ["drive", "test.py"])
        _assert_exit_0(result)
        with open("test.py") as f:
            data = f.read()
        assert "[DRIVE-2004]" in data


def test_detect_config_copy_help():
    from deepdraw.detect.script.config import copy

    _check_help(copy)


def test_dataset_help():
    from deepdraw.binseg.script.dataset import dataset

    _check_help(dataset)


def test_dataset_list_help():
    from deepdraw.binseg.script.dataset import list

    _check_help(list)


def test_dataset_list():
    from deepdraw.binseg.script.dataset import list

    runner = CliRunner()
    result = runner.invoke(list)
    _assert_exit_0(result)
    assert result.output.startswith("Supported datasets:")


def test_dataset_check_help():
    from deepdraw.binseg.script.dataset import check

    _check_help(check)


def test_dataset_check():
    from deepdraw.binseg.script.dataset import check

    runner = CliRunner()
    result = runner.invoke(check, ["--verbose", "--verbose", "--limit=2"])
    _assert_exit_0(result)


def test_detect_dataset_help():
    from deepdraw.detect.script.dataset import dataset

    _check_help(dataset)


def test_detect_dataset_list_help():
    from deepdraw.detect.script.dataset import list

    _check_help(list)


def test_detect_dataset_list():
    from deepdraw.detect.script.dataset import list

    runner = CliRunner()
    result = runner.invoke(list)
    _assert_exit_0(result)
    assert result.output.startswith("Supported datasets:")


def test_detect_dataset_check_help():
    from deepdraw.detect.script.dataset import check

    _check_help(check)


def test_detect_dataset_check():
    from deepdraw.detect.script.dataset import check

    runner = CliRunner()
    result = runner.invoke(check, ["--verbose", "--verbose", "--limit=2"])
    _assert_exit_0(result)