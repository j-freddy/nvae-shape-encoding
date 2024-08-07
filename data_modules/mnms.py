import os
import warnings
from lightning import LightningDataModule
import pandas as pd
import torch
import torchio as tio
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_modules.utils import preprocess
from datasets.mnms import MnMsDataset
from utils.anatomical_validity_checker import AnatomicalValidityChecker
from utils.const import MASK_NUM_CLASSES, MaskClassLabel, MnMs
from utils.utils import listdir

def get_scan_and_mask(
    patient_id: str,
    dir: str,
    frame_ed: str,
    frame_es: str,
) -> tuple[tio.Subject, tio.Subject]:
    def swap_lv_rv_id(mask: torch.Tensor) -> torch.Tensor:
        """
        ACDC dataset uses RV = 1 and LV = 3, but MnMs uses RV = 3 and LV = 1.
        During preprocessing, we swap the IDs to make it consistent with ACDC.
        """
        mask[mask == MaskClassLabel.LV.value] = -1
        mask[mask == MaskClassLabel.RV.value] = MaskClassLabel.LV.value
        mask[mask == -1] = MaskClassLabel.RV.value
        return mask
    
    patient_dir = os.path.join(dir, patient_id)
    
    path_scan = os.path.join(patient_dir, f"{patient_id}_sa.nii.gz")
    path_mask = os.path.join(patient_dir, f"{patient_id}_sa_gt.nii.gz")

    scan = tio.ScalarImage(path_scan)
    mask = tio.LabelMap(path_mask)
    
    scan_ed = tio.ScalarImage(tensor=scan.data[frame_ed].unsqueeze(0))
    mask_ed = tio.LabelMap(tensor=swap_lv_rv_id(mask.data[frame_ed].unsqueeze(0)))
    scan_es = tio.ScalarImage(tensor=scan.data[frame_es].unsqueeze(0))
    mask_es = tio.LabelMap(tensor=swap_lv_rv_id(mask.data[frame_es].unsqueeze(0)))
    
    subject_ed = tio.Subject(
        scan=scan_ed,
        mask=mask_ed,
    )
    
    subject_es = tio.Subject(
        scan=scan_es,
        mask=mask_es,
    )
    
    return subject_ed, subject_es

def get_dataset(dir: str, info_df: pd.DataFrame) -> tio.SubjectsDataset:
    def one_hot(masks: torch.Tensor) -> torch.Tensor:
        masks = torch.squeeze(masks, dim=1)
        masks_onehot = F.one_hot(
            masks.long(),
            num_classes=MASK_NUM_CLASSES
        ).permute(0, 3, 1, 2)
        
        return masks_onehot.float()
    
    def get_anatomical_validity_of_masks(subject: tio.Subject) -> float:
        masks = one_hot(subject.mask.data.permute(3, 0, 1, 2))
        
        num_valid = 0
    
        for mask in masks:
            AV = AnatomicalValidityChecker(mask)
            if AV.count_violations() == 0:
                num_valid += 1
        
        return num_valid / len(masks)
    
    subjects = []

    patient_ids = listdir(dir)
    
    for patient_id in patient_ids:
        subject_ed, subject_es = get_scan_and_mask(
            patient_id,
            dir,
            info_df.loc[patient_id, "ED"],
            info_df.loc[patient_id, "ES"],
        )
        
        subject_ed = preprocess(subject_ed)
        subject_es = preprocess(subject_es)
        
        anatomical_validity_ed = get_anatomical_validity_of_masks(subject_ed)
        anatomical_validity_es = get_anatomical_validity_of_masks(subject_es)
        
        # A subject has same number of slices so we can take geometric mean
        anatomical_validity = (anatomical_validity_ed + anatomical_validity_es) / 2
        
        subject = tio.Subject(
            ed_scan=subject_ed.scan,
            ed_mask=subject_ed.mask,
            es_scan=subject_es.scan,
            es_mask=subject_es.mask,
            height=float(info_df.loc[patient_id, "Height"]),
            weight=float(info_df.loc[patient_id, "Weight"]),
            condition=info_df.loc[patient_id, "Pathology"],
            vendor=info_df.loc[patient_id, "Vendor"],
            centre=int(info_df.loc[patient_id, "Centre"]),
            anatomical_validity_of_masks=anatomical_validity,
        )
        
        subjects.append(subject)
    
    return tio.SubjectsDataset(subjects)

def download_and_preprocess_acdc() -> tuple[tio.SubjectsDataset, tio.SubjectsDataset, tio.SubjectsDataset]:
    info_df = pd.read_csv(MnMs.RAW.INFO_FILE, index_col="External code")

    if os.path.exists(MnMs.TRAIN_PATH):
        print("Preprocessed training data found. Loading...")
        
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            data_train = torch.load(MnMs.TRAIN_PATH)
    else:
        print("Preprocessed training data not found. Preprocessing...")
        
        data_train = get_dataset(MnMs.RAW.TRAIN_PATH_LABELLED, info_df)
        torch.save(data_train, MnMs.TRAIN_PATH)
    
    if os.path.exists(MnMs.VAL_PATH):
        print("Preprocessed validation data found. Loading...")
        
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            data_val = torch.load(MnMs.VAL_PATH)
    else:
        print("Preprocessed validation data not found. Preprocessing...")
        
        data_val = get_dataset(MnMs.RAW.VAL_PATH, info_df)
        torch.save(data_val, MnMs.VAL_PATH)
    
    if os.path.exists(MnMs.TEST_PATH):
        print("Preprocessed test data found. Loading...")
        
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            data_test = torch.load(MnMs.TEST_PATH)
    else:
        print("Preprocessed test data not found. Preprocessing...")
        
        data_test = get_dataset(MnMs.RAW.TEST_PATH, info_df)
        torch.save(data_test, MnMs.TEST_PATH)
    
    return data_train, data_val, data_test

class MnMsDataModule(LightningDataModule):
    """
    Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation
    Challenge (ACDC) data module.
    
    Data is preprocessed to crop to the bounding box around the heart based on
    the provided GT segmentation masks. Each data point is a 4-tuple consisting
    of a single slice with the following information:
    (1) Scan tensor (1x128x128)
    (2) One-hot encoded GT Mask tensor (4x128x128)
    (3) Condition tensor (1)
    (4) Whether the scan is ED/1 or ES/0 (1)

    The condition tensor is an integer with the following indexing:
    - 1: DCM
    - 2: HCM
    - 3: HHD
    - 4: NOR
    - 5: Other
    - 6: ARV
    - 7: AHS
    - 8: IHD
    - 9: LVNC
    """
    
    def __init__(
        self,
        batch_size: int=32,
        filter_empty: bool=False,
        from_vendor: str=None,
        from_centre: int=None,
        num_subjects: int=None,
        sort_by_validity: bool=False,
        augment: bool=False,
        augment_test: bool=False,
    ):
        """
        Args:
            batch_size (int): Batch size. Default: 32.
            filter_empty: Whether to remove slices with empty masks. Default:
                False.
            from_vendor: Use data only from the specified vendor. If None, use
                data from all vendors. Default: None.
            from_centre: Use data only from the specified centre. If None, use
                data from all centres. Default: None.
            num_subjects: Number of subjects to use for both training and
                validation. The subjects are randomly sampled. If None, use all
                available data. Default: None.
            sort_by_validity: If set, instead of randomly sampling subjects,
                prioritise subjects with high percentage of anatomically valid
                masks. Default: False.
            augment: Whether to apply data augmentation on train
                set. Default: False.
            augment_test: Whether to apply data augmentation on
                test set. Default: False.
        """
        super().__init__()
        
        self.batch_size = batch_size
        
        data_train, data_val, data_test = download_and_preprocess_acdc()
        
        if num_subjects is not None:
            # Merge train and validation data together
            data = tio.SubjectsDataset(data_train + data_val)

            data = self._get_data_as_slice(
                data,
                filter_empty,
                from_vendor,
                from_centre,
                num_subjects,
                sort_by_validity,
            )
            
            import sys
            sys.exit()
        
        data_train = self._get_data_as_slice(
            data_train,
            filter_empty,
            from_vendor,
            from_centre,
        )
        
        data_val = self._get_data_as_slice(
            data_val,
            filter_empty,
            from_vendor,
            from_centre,
        )
        
        data_test = self._get_data_as_slice(
            data_test,
            filter_empty,
            from_vendor,
            from_centre,
        )
        
        self.data_train = MnMsDataset(*data_train, augment=augment)
        self.data_val = MnMsDataset(*data_val, augment=False)
        self.data_test = MnMsDataset(*data_test, augment=augment_test)
    
    def _filter_data(
        self,
        data: tio.SubjectsDataset,
        from_vendor: str=None,
        from_centre: int=None,
    ) -> tio.SubjectsDataset:
        """
        Filter by vendor and centre.
        """
        data_filtered = []
        
        for subject in data:
            if from_vendor is not None and subject.vendor != from_vendor:
                continue
            
            if from_centre is not None and subject.centre != from_centre:
                continue
            
            data_filtered.append(subject)
        
        return tio.SubjectsDataset(data_filtered)

    def _sample_data(
        self,
        data: tio.SubjectsDataset,
        num_subjects: int,
        sort_by_validity: bool=False,
    ) -> tio.SubjectsDataset:
        if sort_by_validity:
            data = sorted(
                data,
                key=lambda subject: subject.anatomical_validity_of_masks,
                reverse=True,
            )
            data = data[:num_subjects]
        else:
            idx = torch.randperm(len(data))[:num_subjects]
            data = tio.SubjectsDataset([data[i] for i in idx])
        
        return data
    
    def _get_data_as_slice_from_subject(
        self,
        subject: tio.Subject,
        is_ed: bool=True,
        filter_empty: bool=True,
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
        conditions = MnMs.condition_to_idx[subject.condition] * torch.ones(scans.shape[0])
        
        return scans, masks, conditions
        
    def _get_data_as_slice(
        self,
        data: tio.SubjectsDataset,
        filter_empty: bool=True,
        from_vendor: str=None,
        from_centre: int=None,
        num_subjects: int=None,
        sort_by_validity: bool=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scans = []
        masks = []
        conditions = []
        eds = []
        
        # Filter by vendor and centre
        data = self._filter_data(data, from_vendor, from_centre)
        
        # Sample @num_subjects subjects
        if num_subjects is not None:
            data = self._sample_data(data, num_subjects, sort_by_validity)
        
        for subject in data:
            print(subject.anatomical_validity_of_masks)
        
        import sys
        sys.exit()
        
        # Get data as slice
        for subject in data:
            ed_scans, ed_masks, ed_conditions = self._get_data_as_slice_from_subject(
                subject,
                is_ed=True,
                filter_empty=filter_empty,
            )
            scans.append(ed_scans)
            masks.append(ed_masks)
            conditions.append(ed_conditions)
            eds.append(torch.ones(ed_scans.shape[0]))
            
            es_scans, es_masks, es_conditions = self._get_data_as_slice_from_subject(
                subject,
                is_ed=False,
                filter_empty=filter_empty,
            )
            scans.append(es_scans)
            masks.append(es_masks)
            conditions.append(es_conditions)
            eds.append(torch.zeros(es_scans.shape[0]))
        
        scans = torch.cat(scans)
        masks = self._one_hot(torch.cat(masks))
        conditions = torch.cat(conditions)
        eds = torch.cat(eds)
        
        return scans, masks, conditions, eds

    def _one_hot(self, masks: torch.Tensor) -> torch.Tensor:
        masks = torch.squeeze(masks, dim=1)
        masks_onehot = F.one_hot(
            masks.long(),
            num_classes=MASK_NUM_CLASSES
        ).permute(0, 3, 1, 2)
        
        return masks_onehot.float()

    def train_dataloader(self, shuffle=True):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=shuffle)

    def val_dataloader(self, shuffle=False):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=shuffle)
    
    def test_dataloader(self, shuffle=False):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=shuffle)
