import csv
import os
from lightning import LightningDataModule
import subprocess
import torch
import torchio as tio
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from const import ACDC, DATA_PATH, SCRIPTS_PATH
from dataset.acdc import ACDCMaskDataset

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

def preprocess(subject: tio.Subject) -> tuple[tio.Subject, int]:
    mask = subject.mask.data[0, :, :, :]
    _, _, num_slices = mask.shape

    # :2 to ignore slice index
    nonzero_coords = torch.nonzero(mask)[:, :2]

    # Get bounding box
    min_x = torch.min(nonzero_coords[:, 1]).item()
    max_x = torch.max(nonzero_coords[:, 1]).item()
    min_y = torch.min(nonzero_coords[:, 0]).item()
    max_y = torch.max(nonzero_coords[:, 0]).item()
    
    width = max(max_x - min_x, max_y - min_y)
    padding = 4

    transform = tio.transforms.Compose([
        # Crop to dimensions centred around the mask to minimise background
        tio.CropOrPad(
            (width + padding, width + padding, num_slices),
            mask_name="mask",
        ),
        tio.Resize((128, 128, num_slices)),
        tio.RescaleIntensity((0, 1)),
    ])
    
    return transform(subject), num_slices

def get_dataset(test=False) -> tuple[tio.SubjectsDataset, int]:
    subjects = []
    
    # TODO noqa
    seq = range(101, 151) if test else range(1, 101)
    
    # Small subset to speed up preprocessing
    # seq = range(101, 106) if test else range(1, 6)
    
    max_slices = 0
    
    for i in seq:
        patient_id = str(i).zfill(3)
        
        frame_ed, frame_es = get_frame_ids(patient_id, test)
        subject_ed, subject_es = get_image_and_mask(patient_id, frame_ed, frame_es, test)

        subject_ed, num_slices_ed = preprocess(subject_ed)
        subject_es, num_slices_es = preprocess(subject_es)
        
        max_slices = max(max_slices, num_slices_ed, num_slices_es)
        
        subject = tio.Subject(
            ed_image=subject_ed.image,
            ed_mask=subject_ed.mask,
            es_image=subject_es.image,
            es_mask=subject_es.mask,
        )
        
        subjects.append(subject)
    
    return tio.SubjectsDataset(subjects), max_slices

def download_and_preprocess_acdc() -> tuple[tio.SubjectsDataset, tio.SubjectsDataset, int, int]:
    # Download dataset if not present and preprocessing required
    if not os.path.exists(ACDC.TRAIN_PATH) or not os.path.exists(ACDC.TEST_PATH):
        if not os.path.exists(os.path.join(DATA_PATH, "ACDC")):
            subprocess.run(["sh", os.path.join(SCRIPTS_PATH, "download-acdc.sh")], check=True)

    if os.path.exists(ACDC.TRAIN_PATH):
        print("Preprocessed training data found. Loading...")
        
        d = torch.load(ACDC.TRAIN_PATH)
        data_train = d["data_train"]
        max_slices_train = d["max_slices_train"]
    else:
        print("Preprocessed training data not found. Preprocessing...")
        
        data_train, max_slices_train = get_dataset()
        torch.save({
            "data_train": data_train,
            "max_slices_train": max_slices_train,
        }, ACDC.TRAIN_PATH)
    
    if os.path.exists(ACDC.TEST_PATH):
        print("Preprocessed test data found. Loading...")
        
        d = torch.load(ACDC.TEST_PATH)
        data_test = d["data_test"]
        max_slices_test = d["max_slices_test"]
    else:
        print("Preprocessed test data not found. Preprocessing...")
        
        data_test, max_slices_test = get_dataset(test=True)
        torch.save({
            "data_test": data_test,
            "max_slices_test": max_slices_test,
        }, ACDC.TEST_PATH)
    
    return data_train, data_test, max_slices_train, max_slices_test 

class ACDCDataModule(LightningDataModule):
    """
    Automated Cardiac Diagnosis Challenge (ACDC) dataset.
    
    Data is preprocessed to crop to the bounding box around the heart and
    resized to 128x128xs where s is the number of slices. Intensity values are
    rescaled to [0, 1]. Informataion of each data point is retained, including
    voxel spacing and orientation.
    
    The number of slices may differ per patient (e.g. 10, 8) so each data point
    may not necessarily have the same shape.
    """
    
    def __init__(self, batch_size: int=2):
        super().__init__()
        
        self.batch_size = batch_size
        self.data_train, self.data_test, _, _ = download_and_preprocess_acdc()
    
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False)

class ACDCMaskDataModule(LightningDataModule):
    """
    Automated Cardiac Diagnosis Challenge (ACDC) dataset.
    
    See ACDCDataModule Docstring. This is a more lightweight version where each
    data point only consists of the mask tensor values per slice (128x128x1).
    """
    
    def __init__(
        self,
        batch_size: int=32,
        filter_empty: bool=False,
        one_hot: bool=True,
        register_alignment: bool=False,
        augment: bool=False,
        augment_test: bool=False,
    ):
        super().__init__()
        
        self.batch_size = batch_size
        
        if register_alignment and os.path.exists(ACDC.ALIGNED.TRAIN_PATH) and os.path.exists(ACDC.ALIGNED.TEST_PATH):
            print("Preprocessed aligned masks found. Loading...")
            
            data_train = torch.load(ACDC.ALIGNED.TRAIN_PATH)
            data_test = torch.load(ACDC.ALIGNED.TEST_PATH)
        else:
            data_train, data_test, _, _ = download_and_preprocess_acdc()

            data_train = self._get_masks(data_train, filter_empty, register_alignment)
            # Always preserve empty masks for test set
            data_test = self._get_masks(data_test, filter_empty=False, register_alignment=register_alignment)
            
            # Save aligned masks because it takes a lot of time
            if register_alignment:
                torch.save(data_train, ACDC.ALIGNED.TRAIN_PATH)
                torch.save(data_test, ACDC.ALIGNED.TEST_PATH)
        
        if one_hot:
            data_train = self._one_hot(data_train)
            data_test = self._one_hot(data_test)

        data_train, data_val = self._split_train_val(data_train)
        
        self.data_train = ACDCMaskDataset(data_train, augment)
        self.data_val = ACDCMaskDataset(data_val, augment=False)
        self.data_test = ACDCMaskDataset(data_test, augment=augment_test)
    
    def _register_alignment(self, masks: torch.Tensor) -> torch.Tensor:
        # avg_y is average y-coordinate of right ventricle
        # Align masks so right ventricle is on top
        aligned_masks, best_avg_y = masks, torch.inf
        
        tick_deg = 1
        
        for i in range(0, 360, tick_deg):
            rotated_masks = TF.rotate(masks, i)
            
            # Calculate average y-coordinate of right ventricle (labelled as 1)
            coords = torch.nonzero(rotated_masks[:, 0, :, :] == 1)[:, 1:]
            avg_y = coords[:, 0].float().mean()
            
            if avg_y < best_avg_y:
                best_avg_y = avg_y
                aligned_masks = rotated_masks
        
        return aligned_masks
    
    def _get_masks_from_subject(
        self,
        subject: tio.Subject,
        is_es: bool=False,
        filter_empty: bool=True,
        register_alignment: bool=False,
    ) -> torch.Tensor:
        subject_mask_data = subject.es_mask.data if is_es else subject.ed_mask.data
        
        masks = []
        
        _, _, _, num_slices = subject_mask_data.shape
        
        for slice in range(num_slices):
            mask = subject_mask_data[:, :, :, slice]
            
            if filter_empty and torch.all(mask == 0):
                continue

            masks.append(mask)
        
        masks = torch.stack(masks)
        
        if register_alignment:
            masks = self._register_alignment(masks)
        
        return masks
        
    def _get_masks(
        self,
        data: tio.SubjectsDataset,
        filter_empty: bool=True,
        register_alignment: bool=False,
    ) -> torch.Tensor:
        masks = []
        
        for subject in data:
            ed_masks = self._get_masks_from_subject(
                subject,
                is_es=False,
                filter_empty=filter_empty,
                register_alignment=register_alignment,
            )
            masks.append(ed_masks)
            
            es_masks = self._get_masks_from_subject(
                subject,
                is_es=True,
                filter_empty=filter_empty,
                register_alignment=register_alignment,
            )
            masks.append(es_masks)
        
        return torch.cat(masks)

    def _one_hot(self, masks: torch.Tensor) -> torch.Tensor:
        masks = torch.squeeze(masks, dim=1)
        masks_onehot = F.one_hot(
            masks.long(),
            num_classes=len(masks.unique())
        ).permute(0, 3, 1, 2)
        
        return masks_onehot.float()
    
    def _split_train_val(
        self,
        masks: torch.Tensor,
        perc: float=0.9,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Shuffle masks
        idx = torch.randperm(len(masks))
        masks = masks[idx]
        
        split_idx = int(len(masks) * perc)
        mask_train = masks[:split_idx]
        mask_val = masks[split_idx:]
        
        return mask_train, mask_val
    
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False)
