import os
import os.path as osp
from tqdm import tqdm
from typing import Callable, List, Optional, Union
import numpy as np
import ase, ase.io

import torch
import torch.nn as nn
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import scatter


class MolecularDataset(Dataset):
    '''
    This class is a dataset for molecular data.
    
    Args:
        root (str): The root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in a data object and returns a transformed version. The data object will be transformed before every access. Default: None.
        pre_transform (callable, optional): A function/transform that takes in a data object and returns a transformed version. The data object will be transformed before being saved to disk. Default: None.
        pre_filter (callable, optional): A function that takes in a data object and returns a boolean value, indicating whether the data object should be included in the final dataset. Default: None.
        force_reload (bool): Whether to re-process the dataset. Default: False.
        precision (torch.dtype): The precision of the data. Default: torch.float.
    '''
    def __init__(
        self,
        precision: torch.dtype = torch.float,
        **kwargs,
    ) -> None:
        self.precision = precision
        super().__init__(**kwargs)

        # self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str]]:
        names = [name for name in os.listdir(self.raw_dir) if name.endswith(('.npz', '.xyz', '.extxyz'))]
        return names

    # @property
    def processed_file_names(self) -> List[str]:
        return [name for name in os.listdir(self.processed_dir) if name.startswith('data_') and name.endswith('.pt')]

    def process(self) -> None:
        # data_list = []
        # data_path = self.processed_paths[0]
        idx = 0
        for raw_path in tqdm(self.raw_paths):
            if raw_path.endswith('.npz'):
                # data_list.extend(self.parse_npz(raw_path))
                data_list = self.parse_npz(raw_path)
            elif raw_path.endswith('.xyz') or raw_path.endswith('.extxyz'):
                # data_list.extend(self.parse_xyz(raw_path))
                data_list = self.parse_xyz(raw_path)
            
            for data in data_list:
                torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1
        # self.save(data_list, data_path)

    def len(self) -> int:
        return len(self.processed_file_names())
    
    def get(self, idx: int) -> Data:
        return torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))

    def parse_npz(self, raw_path: str) -> List[Data]:
        data_list = []
        raw_data = np.load(raw_path)

        z = torch.from_numpy(raw_data['Z']).int()
        pos = torch.from_numpy(raw_data['R']).to(self.precision)
        lattice = torch.from_numpy(raw_data['L']).to(self.precision) if 'L' in raw_data else torch.eye(3, dtype=self.precision) * torch.inf
        if lattice.numel() == 3:
            lattice = lattice.diag()
        elif lattice.numel() == 9:
            lattice = lattice.reshape(3, 3)
        else:
            raise ValueError('The lattice must be a single 3x3 matrix for each npz file.')
        energy = torch.from_numpy(raw_data['E']).to(self.precision) if 'E' in raw_data else None
        force = torch.from_numpy(raw_data['F']).to(self.precision) if 'F' in raw_data else None

        for i in range(pos.size(0)):
            data = Data()
            data.z = z.reshape(-1) if z.dim() < 2 else z[i].reshape(-1)
            data.pos = pos[i].reshape(-1, 3)
            data.lattice = lattice.reshape(1, 3, 3)
            if energy is not None:
                data.energy = energy[i].reshape(1)
            if force is not None:
                data.force = force[i].reshape(-1, 3)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list
    
    def parse_xyz(self, raw_path: str) -> List[Data]:
        data_list = []
        atoms_list = ase.io.read(raw_path, index=':')

        for atoms in atoms_list:
            atoms.set_constraint()
            z = torch.from_numpy(atoms.get_atomic_numbers()).int()
            pos = torch.from_numpy(atoms.get_positions()).to(self.precision)
            lattice = torch.from_numpy(atoms.get_cell().array).to(self.precision)
            lattice[lattice.norm(dim=-1) < 1e-3] = torch.inf
            lattice[~atoms.get_pbc()] = torch.inf
            energy = torch.tensor(atoms.get_potential_energy(), dtype=self.precision)
            forces = torch.from_numpy(atoms.get_forces()).to(self.precision)

            data = Data()
            data.z = z.reshape(-1)
            data.pos = pos.reshape(-1, 3)
            data.lattice = lattice.reshape(1, 3, 3)
            data.energy = energy.reshape(1)
            data.force = forces.reshape(-1, 3)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list


class MolecularStatistics(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data):
        stats = {}

        z = data.z.long().cpu()
        z_unique = z.unique()
        stats['z'] = z_unique

        batch = data.batch.cpu()

        stats['properties'] = {}
        try:
            energy = data.energy.cpu()
            formula = scatter(nn.functional.one_hot(z), batch, dim=0).to(energy.dtype)
            solution = torch.linalg.lstsq(formula, energy, driver='gelsd').solution
            energy_shifts = solution[z_unique]
            energy_scale = ((energy - torch.matmul(formula, solution)).square().sum() / (formula).sum()).sqrt()
            stats['properties']['energy'] = {'shift': energy_shifts, 'scale': energy_scale}
        except AttributeError:
            pass
        try:
            force = data.force.norm(dim=-1).cpu()
            force_scale = scatter(force, z, reduce='mean')
            force_scale = force_scale[z_unique]
            stats['properties']['force'] = {'scale': force_scale}
        except AttributeError:
            pass
        return stats
