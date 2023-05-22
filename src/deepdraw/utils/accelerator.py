# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import os

import torch

logger = logging.getLogger(__name__)


class AcceleratorProcessor:
    """This class is used to convert the torch device naming convention to
    lightning's device convention and vice versa.

    It also sets the CUDA_VISIBLE_DEVICES if a gpu accelerator is used.
    """

    def __init__(self, name):
        # Note: "auto" is a valid accelerator in lightning, but there doesn't seem to be a way to check which accelerator it will actually use so we don't take it into account for now.
        self.torch_to_lightning = {"cpu": "cpu", "cuda": "gpu"}

        self.lightning_to_torch = {
            v: k for k, v in self.torch_to_lightning.items()
        }

        self.valid_accelerators = set(
            list(self.torch_to_lightning.keys())
            + list(self.lightning_to_torch.keys())
        )

        self.accelerator, self.device = self._split_accelerator_name(name)

        if self.accelerator not in self.valid_accelerators:
            raise ValueError(f"Unknown accelerator {self.accelerator}")

        # Keep lightning's convention by default
        self.accelerator = self.to_lightning()
        self.setup_accelerator()

    def setup_accelerator(self):
        """If a gpu accelerator is chosen, checks the CUDA_VISIBLE_DEVICES
        environment variable exists or sets its value if specified."""
        if self.accelerator == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA is not currently available, but "
                    f"you set accelerator to '{self.accelerator}'"
                )

            if self.device is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device[0])
            else:
                if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
                    raise ValueError(
                        "Environment variable 'CUDA_VISIBLE_DEVICES' is not set."
                        "Please set 'CUDA_VISIBLE_DEVICES' or specify a device to use, e.g. cuda:0"
                    )
        else:
            # No need to check the CUDA_VISIBLE_DEVICES environment variable if cpu
            pass

        logger.info(
            f"Accelerator set to {self.accelerator} and device to {self.device}"
        )

    def _split_accelerator_name(self, accelerator_name):
        """Splits an accelerator string into accelerator and device components.

        Parameters
        ----------

        accelerator_name: str
            The accelerator (or device in pytorch convention) string (e.g. cuda:0)

        Returns
        -------

        accelerator: str
            The accelerator name
        device: dict[int]
            The selected devices
        """

        split_accelerator = accelerator_name.split(":")
        accelerator = split_accelerator[0]

        if len(split_accelerator) > 1:
            device = split_accelerator[1]
            device = [int(device)]
        else:
            device = None

        return accelerator, device

    def to_torch(self):
        """Converts the accelerator string to torch convention.

        Returns
        -------

        accelerator: str
            The accelerator name in pytorch convention
        """
        if self.accelerator in self.lightning_to_torch:
            return self.lightning_to_torch[self.accelerator]
        elif self.accelerator in self.torch_to_lightning:
            return self.accelerator
        else:
            raise ValueError("Unknown accelerator.")

    def to_lightning(self):
        """Converts the accelerator string to lightning convention.

        Returns
        -------

        accelerator: str
            The accelerator name in lightning convention
        """
        if self.accelerator in self.torch_to_lightning:
            return self.torch_to_lightning[self.accelerator]
        elif self.accelerator in self.lightning_to_torch:
            return self.accelerator
        else:
            raise ValueError("Unknown accelerator.")
