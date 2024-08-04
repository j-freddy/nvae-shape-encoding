import os
import warnings
from lightning import LightningDataModule
import pandas as pd
import torch
import torchio as tio
import torch.nn.functional as F

from data_modules.utils import preprocess
from utils.const import MnMs
from utils.utils import listdir

def get_scan_and_mask(
    patient_id: str,
    dir: str,
    frame_ed: str,
    frame_es: str,
) -> tuple[tio.Subject, tio.Subject]:
    patient_dir = os.path.join(dir, patient_id)
    
    path_scan = os.path.join(patient_dir, f"{patient_id}_sa.nii.gz")
    path_mask = os.path.join(patient_dir, f"{patient_id}_sa_gt.nii.gz")

    scan = tio.ScalarImage(path_scan)
    mask = tio.LabelMap(path_mask)
    
    scan_ed = tio.ScalarImage(tensor=scan.data[frame_ed].unsqueeze(0))
    mask_ed = tio.LabelMap(tensor=mask.data[frame_ed].unsqueeze(0))
    scan_es = tio.ScalarImage(tensor=scan.data[frame_es].unsqueeze(0))
    mask_es = tio.LabelMap(tensor=mask.data[frame_es].unsqueeze(0))
    
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
        
        subject = tio.Subject(
            ed_scan=subject_ed.scan,
            ed_mask=subject_ed.mask,
            es_scan=subject_es.scan,
            es_mask=subject_es.mask,
        )
        
        subjects.append(subject)
    
    return tio.SubjectsDataset(subjects)

def download_and_preprocess_acdc() -> tio.SubjectsDataset:
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
    
    # TODO Other datasets
    return data_train

class MnMsDataModule(LightningDataModule):
    def __init__(self, batch_size: int=32, filter_empty: bool=False):
        super().__init__()
        
        self.batch_size = batch_size
        
        data_train = download_and_preprocess_acdc()
        data_train = self._get_data_as_slice(data_train, filter_empty)
        scans, masks, _, _ = data_train
        print(scans.shape)
        print(masks.shape)
        import sys
        sys.exit()
        
        # self.data_train = ACDCDataset(*data_train, augment=augment)
    
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
        # TODO Update
        # conditions = ACDC.condition_to_idx[subject.condition] * torch.ones(scans.shape[0])
        conditions = 0 * torch.ones(scans.shape[0])
        
        return scans, masks, conditions
        
    def _get_data_as_slice(
        self,
        data: tio.SubjectsDataset,
        filter_empty: bool=True,
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
            num_classes=len(masks.unique())
        ).permute(0, 3, 1, 2)
        
        return masks_onehot.float()
