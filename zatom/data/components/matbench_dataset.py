import json
import os
from typing import Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from zatom.utils.data_utils import hf_download_file
from zatom.utils.typing_utils import typecheck

try:
    from pymatgen.core import Structure

    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False


@typecheck
def get_matbench_data_path(task_name: str) -> str:
    """Download and return path to Matbench data for a given task name.

    Args:
        task_name: Name of the Matbench task.

    Returns:
        Path to the downloaded Matbench dataset.
    """
    path = hf_download_file(
        repo_id="Ty-Perez/matbench_properties",
        filename=f"{task_name}.tar.gz",
    )
    return str(path)


class BaseMatbenchDataset(Dataset):
    """Base class for Matbench datasets.

    Args:
        task_name: Name of the Matbench task.
        fold_idx: Index of the training fold to use.
        split: Data split to use ("train" or "test").
        dataset_dir: Optional path to the dataset directory. If None, the dataset will be downloaded.

    Available Tasks:
        - matbench_dielectric
        - matbench_expt_gap
        - matbench_expt_is_metal
        - matbench_glass
        - matbench_jdft2d
        - matbench_log_gvrh
        - matbench_log_kvrh
        - matbench_mp_e_form
        - matbench_mp_gap
        - matbench_mp_is_metal
        - matbench_perovskites
        - matbench_phonons
        - matbench_steels

    Available Splits:
        - train
        - test
    """

    avail_splits = ["train", "test"]
    avail_tasks = [
        "matbench_dielectric",
        "matbench_expt_gap",
        "matbench_expt_is_metal",
        "matbench_glass",
        "matbench_jdft2d",
        "matbench_log_gvrh",
        "matbench_log_kvrh",
        "matbench_mp_e_form",
        "matbench_mp_gap",
        "matbench_mp_is_metal",
        "matbench_perovskites",
        "matbench_phonons",
        "matbench_steels",
    ]

    @typecheck
    def __init__(
        self,
        task_name: str,
        fold_idx: int = 0,
        split: str = "train",
        dataset_dir: Optional[str] = None,
    ):
        assert (
            task_name in self.avail_tasks
        ), f"Task {task_name} is not one of the available tasks: {self.avail_tasks}"

        if dataset_dir is None:
            dataset_dir = get_matbench_data_path(task_name)

        self.dataset_dir = dataset_dir
        assert os.path.exists(
            dataset_dir
        ), f"Dataset directory for Matbench {dataset_dir} must exist."

        data_files = [f for f in os.listdir(dataset_dir) if f.endswith(".pt")]
        summary_files = [f for f in os.listdir(dataset_dir) if f.endswith(".json")]

        assert len(data_files) > 0, f"No data files found in {dataset_dir}."
        assert (
            len(summary_files) == 1
        ), f"Expected one JSON file in {dataset_dir}, but found {len(summary_files)}. Please remove all JSON files except `summary.json`."

        # Load summary file
        summary_file = os.path.join(dataset_dir, summary_files[0])
        with open(summary_file) as f:
            summary = json.load(f)

        # Collect metadata
        self.mb_task = summary["task_name"]
        self.mb_target_name = summary["target_key"]
        self.metadata = summary["metadata"]
        self.all_sample_ids = summary["all_ids"]
        self.targets_dict = summary["targets"]

        assert len(self.all_sample_ids) == len(
            data_files
        ), f"Number of data files {len(data_files)} does not match number of sample IDs {len(self.all_sample_ids)}."

        # Parse training folds
        fold_idx = str(int(fold_idx))
        self.all_folds = summary["training_folds"]
        avail_folds = list(self.all_folds.keys())

        assert (
            fold_idx in avail_folds
        ), f"fold_idx {fold_idx} not in available folds: {avail_folds}"
        assert (
            split in self.avail_splits
        ), f"split {split} not in available splits: {self.avail_splits}"

        self.split = split
        self.fold_idx = fold_idx
        self.data = self.all_folds[fold_idx][split]

    @typecheck
    def __len__(self) -> int:
        """Return size of dataset.

        Returns:
            Size of the dataset.
        """
        return len(self.data)

    @typecheck
    def info(self) -> str:
        """Return basic information about dataset retrieved from metadata.

        Returns:
            String containing basic information about the dataset.
        """
        out = []
        for key, value in self.metadata.items():
            if key == "bibtex_refs":
                # pass
                continue
            out.append(f"{key}: {value}\n")
        return "".join(out)

    @typecheck
    def __repr__(self) -> str:
        """Return basic information about dataset retrieved from metadata.

        Returns:
            String containing basic information about the dataset.
        """
        out_str = [f"Name: {self.mb_task}\n"]
        out_str += [f"Fold {self.fold_idx}: {self.split}\n"]
        out_str += [f"size: {len(self.data)} / {len(self.all_sample_ids)}\n"]
        return "".join(out_str)

    @typecheck
    def load_sample(self, name: str) -> tuple:
        """Load a single sample from the dataset.

        Args:
            name: Name of the sample to load.

        Returns:
            A tuple containing the structure and target value.
        """
        path = os.path.join(self.dataset_dir, f"{name}.pt")
        sample_dict = torch.load(path, weights_only=True)
        target = sample_dict["target"]
        structure = sample_dict["structure"]
        if PYMATGEN_AVAILABLE:
            structure = Structure.from_dict(sample_dict["structure"])

        return structure, target

    @typecheck
    def __getitem__(self, idx: int) -> tuple:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple containing the structure and target value.
        """
        sample_name = self.data[idx]
        structure, target = self.load_sample(sample_name)
        return structure, target


class PyGMatbenchDataset(BaseMatbenchDataset):
    """Matbench dataset converted to PyG Data objects.

    NOTE: This requires pymatgen to work properly.

    Args:
        transform: Optional transform to apply to each Data object.
        **kwargs: Additional arguments passed to BaseMBDataset.
    """

    @typecheck
    def __init__(self, transform=None, **kwargs):

        super().__init__(**kwargs)
        self.transform = transform

    @typecheck
    def __getitem__(self, idx: int) -> Data:
        """Get a single sample from the dataset as a PyG Data object.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A PyG Data object containing the structure and target value.
        """
        structure, target = super().__getitem__(idx)

        data = Data()
        data.pos = torch.tensor(structure.cart_coords, dtype=torch.float)
        data.z = torch.tensor(structure.atomic_numbers, dtype=torch.long)
        data.cell = torch.tensor(structure.lattice.matrix, dtype=torch.float).reshape(1, 3, 3)
        data.pbc = torch.tensor(structure.lattice.pbc, dtype=torch.bool).reshape(1, 3)
        data.y = torch.tensor([target], dtype=torch.float)

        if self.transform:
            data = self.transform(data)

        return data
