import pydicom
from typing import List
from pathlib import Path


class SeriesDataset:
    """Minimal version of highdicom.reader.SeriesDataset"""

    def __init__(self, dcm_files: List[pydicom.Dataset]):
        self._datasets = dcm_files
        self._instance_map = {
            d.SOPInstanceUID: d for d in self._datasets if 'SOPInstanceUID' in d
        }

    @classmethod
    def from_files(cls, path: str) -> "SeriesDataset":
        """Load all DICOM files in a directory into a SeriesDataset"""
        dcm_files = []

        for file_path in Path(path).glob("*.dcm"):
            try:
                ds = pydicom.dcmread(str(file_path))
                dcm_files.append(ds)
            except Exception as e:
                print(f"⚠️ Skipping {file_path}: {e}")

        if not dcm_files:
            raise ValueError(f"No valid DICOM files found in directory: {path}")

        return cls(dcm_files)

    def __getitem__(self, sop_instance_uid: str) -> pydicom.Dataset:
        return self._instance_map[sop_instance_uid]

    def __len__(self) -> int:
        return len(self._datasets)

    def get_all_instances(self) -> List[pydicom.Dataset]:
        return list(self._datasets)
