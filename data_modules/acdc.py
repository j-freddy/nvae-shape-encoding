import csv
import os
from lightning import LightningDataModule
import subprocess
import torch
import torchio as tio
from torch.utils.data import DataLoader

from const import ACDC, DATA_PATH, SCRIPTS_PATH

def get_frame_ids(patient_id: str, test: bool=False) -> tuple[str, str]:
    info_file = os.path.join(
        ACDC.RAW.TEST_PATH if test else ACDC.RAW.TRAIN_PATH,
        f"patient{patient_id}",
        "Info.cfg",
    )
    
    with open(info_file, "r") as f:
        reader = csv.reader(f, delimiter=":")
        
        ed, frame_ed = next(reader)
        es, frame_es = next(reader)
        
        frame_ed = frame_ed.strip().zfill(2)
        frame_es = frame_es.strip().zfill(2)
        
    assert ed == "ED"
    assert es == "ES"
    
    return frame_ed, frame_es

def get_image_and_mask(
    patient_id: str,
    frame_ed: str,
    frame_es: str,
    test: bool=False,
) -> tuple[tio.Subject, tio.Subject]:
    patient_dir = os.path.join(
        ACDC.RAW.TEST_PATH if test else ACDC.RAW.TRAIN_PATH,
        f"patient{patient_id}",
    )
    
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
        image=image_ed,
        mask=image_ed_gt,
    )
    
    subject_es = tio.Subject(
        image=image_es,
        mask=image_es_gt,
    )
    
    return subject_ed, subject_es

def preprocess(subject: tio.Subject) -> tio.Subject:
    mask = subject.mask.data[0, :, :, :]

    # :2 to ignore slice index
    nonzero_coords = torch.nonzero(mask)[:, :2]

    # Get bounding box
    min_x = torch.min(nonzero_coords[:, 1]).item()
    max_x = torch.max(nonzero_coords[:, 1]).item()
    min_y = torch.min(nonzero_coords[:, 0]).item()
    max_y = torch.max(nonzero_coords[:, 0]).item()
    
    width = max(max_x - min_x, max_y - min_y)

    transform = tio.transforms.Compose([
        # Crop to dimensions centred around the mask to minimise background
        tio.CropOrPad(
            (width, width, 10),
            mask_name="mask",
        ),
        tio.Resize((128, 128, 10)),
        tio.RescaleIntensity((0, 1)),
    ])
    
    return transform(subject)

def get_dataset(test=False) -> tio.SubjectsDataset:
    subjects = []
    
    # TODO noqa
    # seq = range(101, 151) if test else range(1, 101)
    
    # Small subset to speed up preprocessing
    seq = range(101, 106) if test else range(1, 6)
    
    for i in seq:
        patient_id = str(i).zfill(3)
        
        frame_ed, frame_es = get_frame_ids(patient_id, test)
        subject_ed, subject_es = get_image_and_mask(patient_id, frame_ed, frame_es, test)

        subject_ed = preprocess(subject_ed)
        subject_es = preprocess(subject_es)
        
        subject = tio.Subject(
            ed_image=subject_ed.image,
            ed_mask=subject_ed.mask,
            es_image=subject_es.image,
            es_mask=subject_es.mask,
        )
        
        subjects.append(subject)
    
    return tio.SubjectsDataset(subjects)

def download_and_preprocess_acdc() -> tuple[tio.SubjectsDataset, tio.SubjectsDataset]:
    # Download dataset if not present and preprocessing required
    if not os.path.exists(ACDC.TRAIN_PATH) or not os.path.exists(ACDC.TEST_PATH):
        if not os.path.exists(os.path.join(DATA_PATH, "ACDC")):
            subprocess.run(["sh", os.path.join(SCRIPTS_PATH, "download-acdc.sh")], check=True)

    if os.path.exists(ACDC.TRAIN_PATH):
        print("Preprocessed training data found. Loading...")
        data_train = torch.load(ACDC.TRAIN_PATH)
    else:
        print("Preprocessed training data not found. Preprocessing...")
        data_train = get_dataset()
        torch.save(data_train, ACDC.TRAIN_PATH)
    
    if os.path.exists(ACDC.TEST_PATH):
        print("Preprocessed test data found. Loading...")
        data_test = torch.load(ACDC.TEST_PATH)
    else:
        print("Preprocessed test data not found. Preprocessing...")
        data_test = get_dataset(test=True)
        torch.save(data_test, ACDC.TEST_PATH)
    
    return data_train, data_test

class ACDCDataModule(LightningDataModule):
    """
    Automated Cardiac Diagnosis Challenge (ACDC) dataset.
    
    Data is preprocessed to crop to the bounding box around the heart and
    resized to 128x128x10. Intensity values are rescaled to [0, 1]. Informataion
    of each data point is retained, including voxel spacing and orientation.
    """
    
    def __init__(self, batch_size: int=2):
        super().__init__()
        
        self.batch_size = batch_size
        self.data_train, self.data_test = download_and_preprocess_acdc()
    
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)

class ACDCMaskDataModule(LightningDataModule):
    """
    Automated Cardiac Diagnosis Challenge (ACDC) dataset.
    
    See ACDCDataModule Docstring. This is a more lightweight version where each
    data point only consists of the mask tensor values per slice (128x128x1).
    """
    
    def __init__(self, batch_size: int=32):
        super().__init__()
        
        self.batch_size = batch_size
        
        data_train, data_test = download_and_preprocess_acdc()

        self.data_train = self._get_masks(data_train)
        self.data_test = self._get_masks(data_test)
        
    def _get_masks(self, data: tio.SubjectsDataset) -> torch.Tensor:
        num_channels, width, height, num_slices = data[0].ed_mask.data.shape
        
        masks = torch.empty((
            len(data) * num_slices * 2,
            num_channels,
            width,
            height,
        ))
        
        acc = 0 
        
        for subject in data:
            # TODO Add a flag to filter slices with no mask (i.e. 0 everywhere)
            for slice in range(num_slices):
                masks[acc] = subject.ed_mask.data[:, :, :, slice]
                acc += 1
            
            for slice in range(num_slices):
                masks[acc] = subject.es_mask.data[:, :, :, slice]
                acc += 1
        
        return masks
    
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)
