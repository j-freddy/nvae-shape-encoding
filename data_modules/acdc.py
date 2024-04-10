import csv
import os
from lightning import LightningDataModule
import subprocess
import torch
import torchio as tio
from torch.utils.data import DataLoader

from const import ACDC_DATA_PATH, DATA_PATH, SCRIPTS_PATH

class ACDCDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str=DATA_PATH,
        batch_size: int=32,
    ):
        super().__init__()
        
        if not os.path.exists(os.path.join(data_dir, "ACDC")):
            subprocess.run(["sh", os.path.join(SCRIPTS_PATH, "download-acdc.sh")], check=True)
        
        print("Done.")
        
        # TODO
        # python -m data_viewer --dataset acdc
        
        patient_id = "001"
        
        frame_ed, frame_es = self._get_frame_ids(patient_id)
        subject_ed, subject_es = self._get_image_and_mask(patient_id, frame_ed, frame_es)

        mask = subject_ed.ed_mask.data[0, :, :, :]

        # :2 to ignore slice index
        nonzero_coords = torch.nonzero(mask)[:, :2]

        min_x = torch.min(nonzero_coords[:, 1]).item()
        max_x = torch.max(nonzero_coords[:, 1]).item()
        min_y = torch.min(nonzero_coords[:, 0]).item()
        max_y = torch.max(nonzero_coords[:, 0]).item()

        transform = tio.transforms.Compose([
            # Crop to dimensions centred around the mask
            tio.CropOrPad(
                (max_y - min_y, max_x - min_x, 10),
                mask_name="ed_mask",
            ),
            tio.RescaleIntensity((0, 1)),
        ])
        
        subject_ed = transform(subject_ed)
        subject_ed.plot()
        
        import sys
        sys.exit()

        self.data_train = NotImplemented
        self.data_test = NotImplemented
    
    def _get_frame_ids(self, patient_id: str) -> tuple[str, str]:
        info_file = os.path.join(
            ACDC_DATA_PATH,
            f"patient{patient_id}",
            "Info.cfg",
        )
        
        with open(info_file, "r") as f:
            reader = csv.reader(f, delimiter=":")
            
            ed, frame_ed = next(reader)
            es, frame_es = next(reader)
            
            frame_ed = frame_ed.strip()
            frame_es = frame_es.strip()
            
            if len(frame_ed) == 1:
                frame_ed = "0" + frame_ed
            
        assert ed == "ED"
        assert es == "ES"
        
        return frame_ed, frame_es
    
    def _get_image_and_mask(
        self,
        patient_id: str,
        frame_ed: str,
        frame_es: str,
    ) -> tuple[tio.Subject, tio.Subject]:
        patient_dir = os.path.join(ACDC_DATA_PATH, f"patient{patient_id}")
        
        path_ed = os.path.join(
            patient_dir,
            f"patient{patient_id}_frame{frame_ed}.nii.gz",
        )
        
        path_ed_gt = os.path.join(
            patient_dir,
            f"patient{patient_id}_frame{frame_ed}_gt.nii.gz",
        )
        
        path_es = os.path.join(
            patient_dir,
            f"patient{patient_id}_frame{frame_es}.nii.gz",
        )
        
        path_es_gt = os.path.join(
            patient_dir,
            f"patient{patient_id}_frame{frame_es}_gt.nii.gz",
        )
        
        image_ed = tio.ScalarImage(path_ed)
        image_ed_gt = tio.LabelMap(path_ed_gt)
        image_es = tio.ScalarImage(path_es)
        image_es_gt = tio.LabelMap(path_es_gt)
        
        subject_ed = tio.Subject(
            ed_image=image_ed,
            ed_mask=image_ed_gt,
        )
        
        subject_es = tio.Subject(
            es_image=image_es,
            es_mask=image_es_gt,
        )
        
        return subject_ed, subject_es
    
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)
