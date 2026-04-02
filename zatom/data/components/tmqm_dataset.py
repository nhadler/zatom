"""tmQM dataset: transition metal complexes from the Cambridge Structural Database."""

import os
from typing import Callable, List, Literal, Optional

import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from zatom.utils import pylogger
from zatom.utils.typing_utils import typecheck

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

# Numeric property columns from tmQM_y.csv (semicolon-delimited)
TMQM_PROPERTY_COLUMNS = [
    "Electronic_E",
    "Dispersion_E",
    "Dipole_M",
    "Metal_q",
    "HL_Gap",
    "HOMO_Energy",
    "LUMO_Energy",
    "Polarizability",
]

# Map element symbols to atomic numbers
ELEMENT_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
    "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
}


class tmQM(InMemoryDataset):
    """tmQM dataset as a PyG InMemoryDataset.

    Transition metal complexes with DFT-computed properties from the
    Cambridge Structural Database.

    Args:
        root: Root directory where the dataset should be saved.
        transform: A function/transform applied to each Data object on access.
        pre_transform: A function/transform applied before saving to disk.
        pre_filter: A function that filters Data objects before saving.
        load: Whether to load the processed dataset into memory.
        force_reload: Whether to re-process the dataset.
        split: The dataset split to load (train, val, test).
    """

    @typecheck
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        load: bool = True,
        force_reload: bool = False,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        self.split = split
        self.split_dir = os.path.join(root, "raw", split)
        self.processed_file = os.path.join(root, "processed", f"{self.split}.pt")

        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)

        if load:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [self.split]

    @property
    def processed_file_names(self) -> List[str]:
        return [os.path.basename(self.processed_file)]

    def download(self) -> None:
        pass  # User provides raw files

    def process(self) -> None:
        # Load CSV properties for this split
        csv_path = os.path.join(self.split_dir, f"tmQM_y_{self.split}.csv")
        properties = {}
        with open(csv_path, "r") as f:
            header = f.readline().strip().split(";")
            prop_indices = [header.index(col) for col in TMQM_PROPERTY_COLUMNS]
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split(";")
                csd_code = parts[0]
                props = [float(parts[i]) for i in prop_indices]
                properties[csd_code] = props

        # Parse charge and spin from comment line: "CSD_code = X | q = Q | S = S | ..."
        xyz_files = sorted(f for f in os.listdir(self.split_dir) if f.endswith(".xyz"))
        log.info(f"Processing {len(xyz_files)} tmQM {self.split} entries")

        data_list = []
        for fname in tqdm(xyz_files, desc=f"Processing tmQM {self.split}"):
            filepath = os.path.join(self.split_dir, fname)
            csd_code = fname.replace(".xyz", "")

            with open(filepath, "r") as f:
                natoms = int(f.readline().strip())
                comment = f.readline().strip()

                # Parse comment: "CSD_code = X | q = Q | S = S | ..."
                comment_parts = {
                    p.split("=")[0].strip(): p.split("=")[1].strip()
                    for p in comment.split("|")
                    if "=" in p
                }
                charge = int(comment_parts.get("q", "0"))
                spin = int(comment_parts.get("S", "0"))

                atomic_numbers = []
                positions = []
                for _ in range(natoms):
                    parts = f.readline().split()
                    atomic_numbers.append(ELEMENT_TO_Z[parts[0]])
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

            z = torch.tensor(atomic_numbers, dtype=torch.long)
            pos = torch.tensor(positions, dtype=torch.float32)
            N = natoms

            # Build y tensor: 8 tmQM properties padded with NaN to num_global_properties
            props = properties.get(csd_code)
            if props is not None:
                y = torch.tensor([props], dtype=torch.float32)  # (1, 8)
            else:
                y = torch.tensor([[float("nan")] * len(TMQM_PROPERTY_COLUMNS)], dtype=torch.float32)

            pyg_data = Data(
                id=f"tmqm_{csd_code}",
                atom_types=z,
                pos=pos,
                frac_coords=torch.zeros_like(pos),
                cell=torch.zeros((1, 3, 3)),
                lattices=torch.zeros(1, 6),
                lattices_scaled=torch.zeros(1, 6),
                lengths=torch.zeros(1, 3),
                lengths_scaled=torch.zeros(1, 3),
                angles=torch.zeros(1, 3),
                angles_radians=torch.zeros(1, 3),
                num_atoms=torch.LongTensor([N]),
                num_nodes=torch.LongTensor([N]),
                spacegroup=torch.zeros(1, dtype=torch.long),
                token_idx=torch.arange(N),
                dataset_idx=torch.tensor([7], dtype=torch.long),
                y=y,
                charge=torch.tensor(charge, dtype=torch.float32),
                spin=torch.tensor(spin, dtype=torch.long),
            )

            if self.pre_filter is not None and not self.pre_filter(pyg_data):
                continue
            if self.pre_transform is not None:
                pyg_data = self.pre_transform(pyg_data)

            data_list.append(pyg_data)

        log.info(f"Saving {len(data_list)} tmQM {self.split} entries")
        self.save(data_list, self.processed_file)

        # Save num_nodes for all splits into a single file at the dataset root
        # (only done once — whichever split processes first triggers it)
        num_nodes_path = os.path.join(self.root, "num_nodes.pt")
        if not os.path.exists(num_nodes_path):
            all_num_nodes = []
            for split in ("train", "val", "test"):
                split_dir = os.path.join(self.root, "raw", split)
                if not os.path.isdir(split_dir):
                    continue
                for fn in os.listdir(split_dir):
                    if not fn.endswith(".xyz"):
                        continue
                    with open(os.path.join(split_dir, fn), "r") as f:
                        all_num_nodes.append(int(f.readline().strip()))
            torch.save(torch.tensor(all_num_nodes), num_nodes_path)
            log.info(f"Saved num_nodes.pt with {len(all_num_nodes)} entries")
