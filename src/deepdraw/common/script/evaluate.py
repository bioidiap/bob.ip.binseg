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

import logging

logger = logging.getLogger(__name__)


def base_evaluate(
    output_folder,
    predictions_folder,
    dataset,
    second_annotator,
    overlayed,
    threshold,
    steps,
    parallel,
    detection,
    **kwargs,
):
    """Create base evaluate function for segmentation / detection tasks."""
    if detection:
        from ...detect.engine.evaluator import compare_annotators, run
    else:
        from ...binseg.engine.evaluator import compare_annotators, run

    def _validate_threshold(t, dataset):
        """Validate the user threshold selection.

        Returns parsed threshold.
        """
        if t is None:
            return 0.5

        try:
            # we try to convert it to float first
            t = float(t)
            if t < 0.0 or t > 1.0:
                raise ValueError(
                    "Float thresholds must be within range [0.0, 1.0]"
                )
        except ValueError:
            # it is a bit of text - assert dataset with name is available
            if not isinstance(dataset, dict):
                raise ValueError(
                    "Threshold should be a floating-point number "
                    "if your provide only a single dataset for evaluation"
                )
            if t not in dataset:
                raise ValueError(
                    f"Text thresholds should match dataset names, "
                    f"but {t} is not available among the datasets provided ("
                    f"({', '.join(dataset.keys())})"
                )

        return t

    threshold = _validate_threshold(threshold, dataset)

    if not isinstance(dataset, dict):
        dataset = {"test": dataset}

    if second_annotator is None:
        second_annotator = {}
    elif not isinstance(second_annotator, dict):
        second_annotator = {"test": second_annotator}
    # else, second_annotator must be a dict

    if isinstance(threshold, str):
        # first run evaluation for reference dataset, do not save overlays
        logger.info(f"Evaluating threshold on '{threshold}' set")
        threshold = run(
            dataset[threshold], threshold, predictions_folder, steps=steps
        )
        logger.info(f"Set --threshold={threshold:.5f}")

    # clean-up the overlayed path
    if overlayed is not None:
        overlayed = overlayed.strip()

    # now run with the
    for k, v in dataset.items():
        if k.startswith("_"):
            logger.info(f"Skipping dataset '{k}' (not to be evaluated)")
            continue
        logger.info(f"Analyzing '{k}' set...")
        run(
            v,
            k,
            predictions_folder,
            output_folder,
            overlayed,
            threshold,
            steps=steps,
            parallel=parallel,
        )
        second = second_annotator.get(k)
        if second is not None:
            if not second.all_keys_match(v):
                logger.warning(
                    f"Key mismatch between `dataset[{k}]` and "
                    f"`second_annotator[{k}]` - skipping "
                    f"second-annotator comparisons for {k} subset"
                )
            else:
                compare_annotators(
                    v, second, k, output_folder, overlayed, parallel=parallel
                )
