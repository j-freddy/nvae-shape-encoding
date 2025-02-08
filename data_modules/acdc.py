import csv
import os
import warnings
from lightning import LightningDataModule
import subprocess
import torch
import torchio as tio
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from data_modules.utils import preprocess
from utils.const import ACDC, DATA_PATH, SCRIPTS_PATH
from datasets.acdc import ACDC3DDataset, ACDCDataset, ACDCMaskDataset, ACDCWithPredictedMaskDataset, ACDC3DWithPredictedMaskDataset
from utils.utils import one_hot, one_hot_to_image

def get_info(patient_id: str, test: bool=False) -> dict:
    info_file = os.path.join(
        ACDC.RAW.TEST_PATH if test else ACDC.RAW.TRAIN_PATH,
        f"patient{patient_id}",
        "Info.cfg",
    )
    
    with open(info_file, "r") as f:
        reader = csv.reader(f, delimiter=":")
        info = {key: value.strip() for key, value in reader}
        info["Height"] = float(info["Height"])
        info["Weight"] = float(info["Weight"])
        
    return info

def get_scan_and_mask(
    patient_id: str,
    frame_ed: str,
    frame_es: str,
    test: bool=False,
) -> tuple[tio.Subject, tio.Subject]:
    patient_dir = os.path.join(
        ACDC.RAW.TEST_PATH if test else ACDC.RAW.TRAIN_PATH,
        f"patient{patient_id}",
    )
    
    path_scan_ed = os.path.join(
        patient_dir,
        f"patient{patient_id}_frame{frame_ed}.nii.gz",
    )
    
    path_mask_ed = os.path.join(
        patient_dir,
        f"patient{patient_id}_frame{frame_ed}_gt.nii.gz",
    )
    
    path_scan_es = os.path.join(
        patient_dir,
        f"patient{patient_id}_frame{frame_es}.nii.gz",
    )
    
    path_mask_es = os.path.join(
        patient_dir,
        f"patient{patient_id}_frame{frame_es}_gt.nii.gz",
    )
    
    scan_ed = tio.ScalarImage(path_scan_ed)
    mask_ed = tio.LabelMap(path_mask_ed)
    scan_es = tio.ScalarImage(path_scan_es)
    mask_es = tio.LabelMap(path_mask_es)
    
    subject_ed = tio.Subject(
        scan=scan_ed,
        mask=mask_ed,
    )
    
    subject_es = tio.Subject(
        scan=scan_es,
        mask=mask_es,
    )
    
    return subject_ed, subject_es

def get_dataset(test=False) -> tio.SubjectsDataset:
    subjects = []

    seq = range(101, 151) if test else range(1, 101)
    
    # Small subset to speed up preprocessing
    # seq = range(101, 106) if test else range(1, 6)
    
    for i in seq:
        patient_id = str(i).zfill(3)
        info = get_info(patient_id, test)
        
        frame_ed = info["ED"].zfill(2)
        frame_es = info["ES"].zfill(2)

        subject_ed, subject_es = get_scan_and_mask(patient_id, frame_ed, frame_es, test)

        subject_ed = preprocess(subject_ed)
        subject_es = preprocess(subject_es)
        
        subject = tio.Subject(
            ed_scan=subject_ed.scan,
            ed_mask=subject_ed.mask,
            es_scan=subject_es.scan,
            es_mask=subject_es.mask,
            height=info["Height"],
            weight=info["Weight"],
            condition=info["Group"],
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
        
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            data_train = torch.load(ACDC.TRAIN_PATH)
    else:
        print("Preprocessed training data not found. Preprocessing...")
        
        data_train = get_dataset()
        torch.save(data_train, ACDC.TRAIN_PATH)
    
    if os.path.exists(ACDC.TEST_PATH):
        print("Preprocessed test data found. Loading...")
        
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            data_test = torch.load(ACDC.TEST_PATH)
    else:
        print("Preprocessed test data not found. Preprocessing...")
        
        data_test = get_dataset(test=True)
        torch.save(data_test, ACDC.TEST_PATH)
    
    return data_train, data_test

class ACDCDataModule(LightningDataModule):
    """
    Automated Cardiac Diagnosis Challenge (ACDC) data module.
    
    Data is preprocessed to crop to the bounding box around the heart based on
    the provided GT segmentation masks. Each data point is a 4-tuple consisting
    of a single slice with the following information:
    (1) Scan tensor (1x128x128)
    (2) One-hot encoded GT Mask tensor (4x128x128)
    (3) Condition tensor (1)
    (4) Whether the scan is ED/1 or ES/0 (1)

    The condition tensor is an integer with the following indexing:
    - 1: NOR
    - 2: MINF
    - 3: DCM
    - 4: HCM
    - 5: RV
    """
    
    def __init__(
        self,
        batch_size: int=32,
        filter_empty: bool=False,
        register_alignment: bool=False,
        augment: bool=False,
        augment_test: bool=False,
    ):
        """
        Args:
            batch_size (int): Batch size. Default: 32.
            filter_empty: Whether to remove slices with empty masks. Default:
                False.
            register_alignment: Whether to register at slice level such that RV
                points upwards. Default: False.
            augment: Whether to apply data augmentation on train set. Default:
                False.
            augment_test: Whether to apply data augmentation on test set.
                Default: False.
        """
        super().__init__()
        
        self.batch_size = batch_size
        
        if register_alignment and os.path.exists(ACDC.ALIGNED.TRAIN_PATH)\
            and os.path.exists(ACDC.ALIGNED.TEST_PATH):

            print("Preprocessed aligned masks found. Loading...")
            
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                data_train = torch.load(ACDC.ALIGNED.TRAIN_PATH)
                data_test = torch.load(ACDC.ALIGNED.TEST_PATH)
        else:
            data_train, data_test = download_and_preprocess_acdc()
            
            data_train = self._get_data_as_slice(data_train, filter_empty, register_alignment)
            # Always preserve empty masks for test set
            data_test = self._get_data_as_slice(data_test, filter_empty=False, register_alignment=register_alignment) 
            
            # Save aligned masks as it takes a lot of time to preprocess
            if register_alignment:
                torch.save(data_train, ACDC.ALIGNED.TRAIN_PATH)
                torch.save(data_test, ACDC.ALIGNED.TEST_PATH)
        
        data_train, data_val = self._split_train_val(data_train)
        
        self.data_train_raw = data_train
        self.data_val_raw = data_val
        self.data_test_raw = data_test
        
        self.data_train = ACDCDataset(*data_train, augment=augment)
        self.data_val = ACDCDataset(*data_val, augment=False)
        self.data_test = ACDCDataset(*data_test, augment=augment_test)
    
    def _register_alignment(
        self,
        scans: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # avg_y is average y-coordinate of right ventricle
        # Align masks so right ventricle is on top
        aligned_scans, aligned_masks, best_avg_y = scans, masks, torch.inf
        
        best_i = 0
        tick_deg = 1
        
        for i in range(0, 360, tick_deg):
            rotated_masks = TF.rotate(masks, i)
            
            # Calculate average y-coordinate of right ventricle (labelled as 1)
            coords = torch.nonzero(rotated_masks[:, 0, :, :] == 1)[:, 1:]
            avg_y = coords[:, 0].float().mean()
            
            if best_avg_y > avg_y:
                aligned_masks = rotated_masks
                best_avg_y = avg_y
                best_i = i
        
        aligned_scans = TF.rotate(scans, best_i)
        
        return aligned_scans, aligned_masks
        
    def _get_data_as_slice_from_subject(
        self,
        subject: tio.Subject,
        is_ed: bool=True,
        filter_empty: bool=True,
        register_alignment: bool=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        subject_scan_data = subject.ed_scan.data if is_ed else subject.es_scan.data
        subject_mask_data = subject.ed_mask.data if is_ed else subject.es_mask.data
        
        scans = []
        masks = []
        
        assert subject_scan_data.shape == subject_mask_data.shape
        _, _, _, num_slices = subject_scan_data.shape
        
        for slice in range(num_slices):
            scan = subject_scan_data[:, :, :, slice]
            mask = subject_mask_data[:, :, :, slice]
            
            if filter_empty and torch.all(mask == 0):
                continue

            scans.append(scan)
            masks.append(mask)
        
        scans = torch.stack(scans)
        masks = torch.stack(masks)
        conditions = ACDC.condition_to_idx[subject.condition] * torch.ones(scans.shape[0])
        
        if register_alignment:
            scans, masks = self._register_alignment(scans, masks)
        
        return scans, masks, conditions
        
    def _get_data_as_slice(
        self,
        data: tio.SubjectsDataset,
        filter_empty: bool=True,
        register_alignment: bool=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scans = []
        masks = []
        conditions = []
        eds = []
        
        for subject in data:
            ed_scans, ed_masks, ed_conditions = self._get_data_as_slice_from_subject(
                subject,
                is_ed=True,
                filter_empty=filter_empty,
                register_alignment=register_alignment,
            )
            scans.append(ed_scans)
            masks.append(ed_masks)
            conditions.append(ed_conditions)
            eds.append(torch.ones(ed_scans.shape[0]))
            
            es_scans, es_masks, es_conditions = self._get_data_as_slice_from_subject(
                subject,
                is_ed=False,
                filter_empty=filter_empty,
                register_alignment=register_alignment,
            )
            scans.append(es_scans)
            masks.append(es_masks)
            conditions.append(es_conditions)
            eds.append(torch.zeros(es_scans.shape[0]))
        
        scans = torch.cat(scans)
        masks = one_hot(torch.cat(masks)).float()
        conditions = torch.cat(conditions)
        eds = torch.cat(eds)
        
        return scans, masks, conditions, eds
    
    def _split_train_val(
        self,
        data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        perc: float=0.9,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Shuffle data
        idx = torch.randperm(len(data[0]))
        data = [d[idx] for d in data]
        
        # Split data
        split_idx = int(len(data[0]) * perc)
        data_train = [d[:split_idx] for d in data]
        data_val = [d[split_idx:] for d in data]
        
        return data_train, data_val
    
    def train_dataloader(self, shuffle=True):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=shuffle)

    def val_dataloader(self, shuffle=False):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=shuffle)
    
    def test_dataloader(self, shuffle=False):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=shuffle)

class ACDCMaskDataModule(LightningDataModule):
    """
    Automated Cardiac Diagnosis Challenge (ACDC) masks data module.
    
    See ACDCDataModule docstring. This is a more lightweight version where each
    data point only consists of the one-hot mask tensor values per slice
    (4x128x128).
    
    Rotation augmentation and registration alignment was experimented with
    extensively for training VAE and NVAE, but was not found to have noticeable
    improvements. They remain as options for this data module.
    
    SimCLR augmentation is used for training SimCLR. It should also be applied
    during validation. Testing does not exist for pretraining and
    @augment_simclr_test should only be used to preview the data in the data
    viewer. For SimCLR only, intensity values are rescaled to [-1, 1] instead of
    [0, 1] as per the norm. See ACDCMaskDataset docstring for more details on
    that regard.
    """
    
    def __init__(
        self,
        batch_size: int=32,
        filter_empty: bool=False,
        register_alignment: bool=False,
        as_image: bool=False,
        augment_rotation: bool=False,
        augment_rotation_test: bool=False,
        augment_simclr: bool=False,
        augment_simclr_test: bool=False,
        return_original: bool=False,
    ):
        """
        Args:
            batch_size (int): Batch size. Default: 32.
            filter_empty: Whether to remove slices with empty masks. Default:
                False.
            register_alignment: Whether to register at slice level such that RV
                points upwards. Default: False.
            as_image: By default, masks are one-hot encoded as 4x128x128. If
                set, the masks are converted to RGB images (3x128x128) where RV,
                MYO, LV correspond to red, green, blue respectively. Default:
                False.
            augment_rotation: Whether to apply rotation augmentation on train
                set. Default: False.
            augment_rotation_test: Whether to apply rotation augmentation on
                test set. Default: False.
            augment_simclr: Whether to apply SimCLR augmentation pipeline on
                train set. Default: False.
            augment_simclr_test: Whether to apply SimCLR augmentation pipeline
                on test set. Default: False.
            return_original: If set, return the original mask as well as the
                augmented pair. Default: False.
        """
        assert not (augment_rotation and augment_simclr)
        assert not (augment_rotation_test and augment_simclr_test)
        
        if augment_simclr:
            assert as_image
        
        super().__init__()
        
        self.batch_size = batch_size
        
        # Get the full ACDC data module with scans, masks and conditions
        data_module = ACDCDataModule(batch_size, filter_empty, register_alignment)
        
        # Extract masks from the data module
        _, data_train, _, _ = data_module.data_train_raw
        _, data_val, _, _ = data_module.data_val_raw
        _, data_test, _, _ = data_module.data_test_raw

        if as_image:
            # Remove background class
            data_train = one_hot_to_image(data_train)
            data_test = one_hot_to_image(data_test)
        
        self.data_train = ACDCMaskDataset(data_train, augment_rotation, augment_simclr, return_original)
        self.data_val = ACDCMaskDataset(data_val, augment_rotation, augment_simclr, return_original)
        self.data_test = ACDCMaskDataset(data_test, augment_rotation_test, augment_simclr_test, return_original)
    
    def train_dataloader(self, shuffle=True):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=shuffle)

    def val_dataloader(self, shuffle=False):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=shuffle)
    
    def test_dataloader(self, shuffle=False):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=shuffle)

class ACDCWithPredictedMaskDataModule(LightningDataModule):
    """
    Automated Cardiac Diagnosis Challenge (ACDC) data module.
    
    This is equivalent to ACDCDataModule, but additionally includes the predicted segmentation from a segmentation model.
    
    For test data, see ACDC3DWithPredictedMaskDataModule.
    
    Each data point is a 5-tuple consisting
    of a single slice with the following information:
    (1) Scan tensor (1x128x128)
    (2) One-hot encoded GT Mask tensor (4x128x128)
    (3) One-hot encoded predicted mask tensor (4x128x128)
    (4) Condition tensor (1)
    (5) Whether the scan is ED/1 or ES/0 (1)

    The condition tensor is an integer with the following indexing:
    - 1: NOR
    - 2: MINF
    - 3: DCM
    - 4: HCM
    - 5: RV
    """
    
    def __init__(
        self,
        batch_size: int=32,
        augment: bool=False,
        augment_test: bool=False,
    ):
        """
        Args:
            batch_size (int): Batch size. Default: 32.
            augment: Whether to apply data augmentation on train set. Default:
                False.
            augment_test: Whether to apply data augmentation on test set.
                Default: False.
        """
        super().__init__()
        
        self.batch_size = batch_size
        
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
        
            data_train = torch.load(os.path.join(DATA_PATH, "acdc_processed_with_predicted_segmentation_train.pt"))
            data_val = torch.load(os.path.join(DATA_PATH, "acdc_processed_with_predicted_segmentation_val.pt"))
        
        self.data_train_raw = data_train
        self.data_val_raw = data_val
        
        self.data_train = ACDCWithPredictedMaskDataset(*data_train, augment=augment)
        self.data_val = ACDCWithPredictedMaskDataset(*data_val, augment=False)
    
    def train_dataloader(self, shuffle=True):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=shuffle)

    def val_dataloader(self, shuffle=False):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=shuffle)

    def test_dataloader(self):
        raise NotImplementedError("Use ACDC3DWithPredictedMaskDataModule for testing.")

class ACDC3DDataModule(LightningDataModule):
    """
    Automated Cardiac Diagnosis Challenge (ACDC) 3D data module.
    
    This is only used during testing to evaluate the 3D DSC metric, and thus
    the training and validation set is not implemented.
    """
    
    def __init__(self):
        super().__init__()
        
        self.batch_size = 1
        
        _, data_test = download_and_preprocess_acdc()
        data_test = self._get_data_as_volume(data_test)
        
        self.data_test = ACDC3DDataset(*data_test)
        
    def _get_data_as_volume(
        self,
        data: tio.SubjectsDataset,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int], list[int]]:
        scans = []
        masks = []
        conditions = []
        eds = []
        
        for subject in data:
            # Unprocessed ACDC is [C, H, W, S] and we want [S, C, H, W]
            # where S is the number of slices
            
            # ED
            subject_scan_data = subject.ed_scan.data.permute(3, 0, 1, 2)
            subject_mask_data = one_hot(subject.ed_mask.data.permute(3, 0, 1, 2)).float()
            condition = ACDC.condition_to_idx[subject.condition]
            
            scans.append(subject_scan_data)
            masks.append(subject_mask_data)
            conditions.append(condition)
            eds.append(1)
            
            # ES
            subject_scan_data = subject.es_scan.data.permute(3, 0, 1, 2)
            subject_mask_data = one_hot(subject.es_mask.data.permute(3, 0, 1, 2)).float()
            
            scans.append(subject_scan_data)
            masks.append(subject_mask_data)
            conditions.append(condition)
            eds.append(0)
        
        return scans, masks, conditions, eds
    
    def train_dataloader(self):
        raise NotImplementedError("ACDC3DDataModule is only used for testing")

    def val_dataloader(self):
        raise NotImplementedError("ACDC3DDataModule is only used for testing")
    
    def test_dataloader(self, shuffle=False):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=shuffle)
class ACDC3DWithPredictedMaskDataModule(LightningDataModule):
    """
    Automated Cardiac Diagnosis Challenge (ACDC) 3D data module with predicted
    mask.
    
    This is only used during testing to evaluate the 3D DSC metric, and thus the
    training and validation set is not implemented.
    """
    
    def __init__(self):
        super().__init__()
        
        self.batch_size = 1
        
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            data_test = torch.load(os.path.join(DATA_PATH, "acdc_processed_with_predicted_segmentation_test.pt"))
        
        self.data_test = ACDC3DWithPredictedMaskDataset(*data_test)
    
    def train_dataloader(self):
        raise NotImplementedError("ACDC3DWithPredictedMaskDataModule is only used for testing")

    def val_dataloader(self):
        raise NotImplementedError("ACDC3DWithPredictedMaskDataModule is only used for testing")
    
    def test_dataloader(self, shuffle=False):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=shuffle)
