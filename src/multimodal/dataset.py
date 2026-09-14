import os
import random
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from tqdm import tqdm
from pydicom.pixel_data_handlers.util import apply_voi_lut


def random_gamma_contrast(img, gamma_range=(0.98, 1.02), contrast_range=(0.95, 1.05)):
    g = random.uniform(*gamma_range)
    img = img.clamp(0, 1) ** g
    c = random.uniform(*contrast_range)
    mean = img.mean()
    img = (img - mean) * c + mean
    return img.clamp(0, 1)


def add_gaussian_noise(img, std_range=(0.0, 0.01)):
    std = random.uniform(*std_range)
    if std > 0:
        noise = torch.randn_like(img) * std
        img = img + noise
    return img.clamp(0, 1)


CLINICAL_SETS = {
    "lumbar_only": [
        "Sex",
        "Age",
        "Lumbar_Cobb",
        "L1_S1_Lordosis",
    ],
    "global_lumbar": [
        "Sex",
        "Age",
        "Cobb",
        "Lumbar_Cobb",
        "L1_S1_Lordosis",
    ],
    "full_spine": [
        "Sex",
        "Age",
        "Cobb",
        "Thoracic_Cobb",
        "T1_12_Kyphosis",
        "T4_12_Kyphosis",
        "Lumbar_Cobb",
        "L1_S1_Lordosis",
    ],
}


class ScoliosisLumbarDataset(Dataset):
    def __init__(
        self,
        data_dir,
        label_csv,
        img_size=512,
        mode="train",
        clinical_set="global_lumbar",
        use_cache=False,
        cache_dir="cache",
        save_images=False,
        save_dir="experiments_lumbar",
    ):
        self.data_dir = data_dir
        self.pa_dir = os.path.join(data_dir, "PA")
        self.lat_dir = os.path.join(data_dir, "LAT")
        self.label_csv = pd.read_csv(label_csv)

        self.img_size = img_size
        self.mode = mode
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.clinical_set = clinical_set

        if clinical_set not in CLINICAL_SETS:
            raise ValueError(
                f"Unknown clinical_set: {clinical_set}. "
                f"Available: {list(CLINICAL_SETS.keys())}"
            )
        self.clinical_cols = CLINICAL_SETS[clinical_set]

        self.save_images = save_images
        self.save_dir = save_dir
        self.preprocessed_dir = os.path.join(save_dir, "preprocessed")
        self.transformed_dir = os.path.join(save_dir, "transformed")
        self._saved_ids = set()

        if self.save_images:
            os.makedirs(self.preprocessed_dir, exist_ok=True)
            os.makedirs(self.transformed_dir, exist_ok=True)

        self._prepare_labels()
        self.pa_files = self._build_file_dict(self.pa_dir)
        self.lat_files = self._build_file_dict(self.lat_dir)
        self._match_available_patients()

        self.preprocessed_data = {}
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            dataset_name = os.path.basename(os.path.normpath(data_dir))
            self.cache_file = os.path.join(
                self.cache_dir,
                f"scoliosis_lumbar_{dataset_name}_{img_size}.pkl"
            )
            self._load_or_create_cache()

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Lambda(lambda x: random_gamma_contrast(
                x, gamma_range=(0.98, 1.02), contrast_range=(0.95, 1.05)
            )),
            transforms.Lambda(lambda x: add_gaussian_noise(x, std_range=(0.0, 0.01))),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def _prepare_labels(self):
        # Handle Sex
        if "Sex" in self.label_csv.columns:
            self.label_csv["Sex"] = self.label_csv["Sex"].map({
                "F": 0, "M": 1, "f": 0, "m": 1
            })
            self.label_csv["Sex"] = self.label_csv["Sex"].fillna(0).astype(float)

        # Normalize patient_id
        if "patient_id" not in self.label_csv.columns:
            if "ID" in self.label_csv.columns:
                self.label_csv["patient_id"] = (
                    self.label_csv["ID"]
                    .astype(str)
                    .str.extract(r"(\d+)")[0]
                    .str.zfill(8)
                )
            else:
                raise ValueError("label_csv must contain a 'patient_id' or 'ID' column.")
        else:
            self.label_csv["patient_id"] = (
                self.label_csv["patient_id"]
                .astype(str)
                .str.extract(r"(\d+)")[0]
                .str.zfill(8)
            )

        required_cols = ["Structural_L"] + self.clinical_cols
        missing_cols = [col for col in required_cols if col not in self.label_csv.columns]
        if missing_cols:
            raise ValueError(f"label_csv is missing required columns: {missing_cols}")

        # Handle missing values in numeric columns
        numeric_cols = []
        for col in self.clinical_cols + ["Structural_L"]:
            if col != "Sex":
                numeric_cols.append(col)

        for col in numeric_cols:
            self.label_csv[col] = self.label_csv[col].fillna(0.0).astype(float)

        self.label_csv["Structural_L"] = self.label_csv["Structural_L"].astype(float)

    def _build_file_dict(self, dicom_dir):
        file_dict = {}
        if not os.path.isdir(dicom_dir):
            return file_dict

        for f in os.listdir(dicom_dir):
            if f.lower().endswith(".dcm"):
                patient_id = Path(f).stem[:8]
                file_dict[patient_id] = os.path.join(dicom_dir, f)
        return file_dict

    def _match_available_patients(self):
        available_ids = set(self.pa_files.keys()) & set(self.lat_files.keys())
        before = len(self.label_csv)
        self.label_csv = (
            self.label_csv[self.label_csv["patient_id"].isin(available_ids)]
            .copy()
            .reset_index(drop=True)
        )
        print(f"[ScoliosisLumbarDataset] Matched PA/LAT patients: {before} -> {len(self.label_csv)}")

    def _load_or_create_cache(self):
        if os.path.exists(self.cache_file):
            print(f"[Cache] Loading existing cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                self.preprocessed_data = pickle.load(f)
            print(f"[Cache] Loaded {len(self.preprocessed_data)} samples!")
        else:
            print(f"[Cache] No cache file found. Creating a new one: {self.cache_file}")
            self._create_cache()
            print("[Cache] Cache saved!")

    def _create_cache(self):
        for idx in tqdm(range(len(self.label_csv)), desc="Preprocessing data for Cache"):
            row = self.label_csv.iloc[idx]
            patient_id = row["patient_id"]

            pa_path = self.pa_files[patient_id]
            lat_path = self.lat_files[patient_id]

            pa_img = self._preprocess_dicom(pa_path)
            lat_img = self._preprocess_dicom(lat_path)

            self.preprocessed_data[patient_id] = {
                "pa": pa_img,
                "lat": lat_img,
            }

        with open(self.cache_file, "wb") as f:
            pickle.dump(self.preprocessed_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.label_csv)

    def __getitem__(self, index):
        row = self.label_csv.iloc[index]
        patient_id = row["patient_id"]

        if self.use_cache and patient_id in self.preprocessed_data:
            pa_img = self.preprocessed_data[patient_id]["pa"]
            lat_img = self.preprocessed_data[patient_id]["lat"]
        else:
            pa_img = self._preprocess_dicom(self.pa_files[patient_id])
            lat_img = self._preprocess_dicom(self.lat_files[patient_id])

        pa_tensor = self._apply_transform(pa_img)
        lat_tensor = self._apply_transform(lat_img)

        if self.save_images and patient_id not in self._saved_ids:
            self._save_images(patient_id, pa_img, lat_img, pa_tensor, lat_tensor)
            self._saved_ids.add(patient_id)

        clinical = torch.tensor(
            [row[col] for col in self.clinical_cols],
            dtype=torch.float32
        )

        label = torch.tensor(row["Structural_L"], dtype=torch.float32)

        return {
            "pa": pa_tensor,
            "lat": lat_tensor,
            "clinical": clinical,
            "patient_id": patient_id,
        }, label

    def get_clinical_dim(self):
        return len(self.clinical_cols)

    def get_clinical_columns(self):
        return list(self.clinical_cols)

    def _read_dicom_image(self, path):
        dicom_file = pydicom.dcmread(path, force=True)
        pixel_array = dicom_file.pixel_array.astype(np.float32)
        slope = float(getattr(dicom_file, "RescaleSlope", 1.0))
        intercept = float(getattr(dicom_file, "RescaleIntercept", 0.0))
        return dicom_file, pixel_array * slope + intercept

    def _apply_windowing(self, dicom_file, pixel_array):
        out = pixel_array.copy()
        try:
            out = apply_voi_lut(out, dicom_file).astype(np.float32)
        except Exception:
            pass
        if getattr(dicom_file, "PhotometricInterpretation", "") == "MONOCHROME1":
            out = out.max() - out
        return out

    def _percentile_clip(self, pixel_array, lower=1, upper=99):
        p_low, p_high = np.percentile(pixel_array, lower), np.percentile(pixel_array, upper)
        return np.clip(pixel_array, p_low, p_high) if p_high > p_low else pixel_array

    def _normalize_minmax(self, pixel_array):
        pixel_array = pixel_array.astype(np.float32)
        vmin, vmax = pixel_array.min(), pixel_array.max()
        return (pixel_array - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(pixel_array)

    def _apply_clahe(self, pixel_array, clip_limit=2.0, tile_grid_size=(8, 8)):
        tmp_u8 = (self._normalize_minmax(pixel_array) * 255).astype(np.uint8)
        return (
            cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            .apply(tmp_u8)
            .astype(np.float32) / 255.0
        )

    def _apply_gaussian_blur(self, pixel_array, ksize=(3, 3), sigma=0.5):
        return cv2.GaussianBlur(pixel_array, ksize, sigmaX=sigma)

    def _pad_to_square(self, pixel_array, pad_value=0):
        h, w = pixel_array.shape
        if h == w:
            return pixel_array

        if h > w:
            pad_left = (h - w) // 2
            pad_right = (h - w) - pad_left
            return np.pad(
                pixel_array,
                ((0, 0), (pad_left, pad_right)),
                mode="constant",
                constant_values=pad_value
            )
        else:
            pad_top = (w - h) // 2
            pad_bottom = (w - h) - pad_top
            return np.pad(
                pixel_array,
                ((pad_top, pad_bottom), (0, 0)),
                mode="constant",
                constant_values=pad_value
            )

    def _resize_square(self, pixel_array, size):
        return cv2.resize(pixel_array, (size, size), interpolation=cv2.INTER_AREA)

    def _preprocess_dicom(self, dcm_path):
        dicom_file, pixel_array = self._read_dicom_image(dcm_path)
        pixel_array = self._apply_windowing(dicom_file, pixel_array)
        pixel_array = self._percentile_clip(pixel_array, lower=1, upper=99)
        pixel_array = self._apply_clahe(pixel_array, clip_limit=2.0, tile_grid_size=(8, 8))
        pixel_array = self._apply_gaussian_blur(pixel_array, ksize=(3, 3), sigma=0.5)
        pixel_array = self._normalize_minmax(pixel_array)
        pixel_array = self._pad_to_square(pixel_array, pad_value=0)
        pixel_array = self._resize_square(pixel_array, self.img_size)
        return (pixel_array * 255).astype(np.uint8)

    def _apply_transform(self, img):
        pil_img = Image.fromarray(img)
        transform = self.train_transform if self.mode == "train" else self.test_transform
        return transform(pil_img)

    def _save_images(self, patient_id, pa_img, lat_img, pa_tensor, lat_tensor):
        prep_patient_dir = os.path.join(self.preprocessed_dir, patient_id)
        tf_patient_dir = os.path.join(self.transformed_dir, patient_id)
        os.makedirs(prep_patient_dir, exist_ok=True)
        os.makedirs(tf_patient_dir, exist_ok=True)

        Image.fromarray(pa_img).save(os.path.join(prep_patient_dir, "PA_prep.png"))
        Image.fromarray(lat_img).save(os.path.join(prep_patient_dir, "LAT_prep.png"))

        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)

        pa_vis = torch.clamp(pa_tensor * std + mean, 0, 1)
        lat_vis = torch.clamp(lat_tensor * std + mean, 0, 1)

        to_pil_image(pa_vis).save(os.path.join(tf_patient_dir, "PA_tf.png"))
        to_pil_image(lat_vis).save(os.path.join(tf_patient_dir, "LAT_tf.png"))


def custom_collate_fn(batch):
    return {
        "pa": torch.stack([b[0]["pa"] for b in batch]),
        "lat": torch.stack([b[0]["lat"] for b in batch]),
        "clinical": torch.stack([b[0]["clinical"] for b in batch]),
        "patient_id": [b[0]["patient_id"] for b in batch],
    }, torch.stack([b[1] for b in batch])


if __name__ == "__main__":
    dataset_dir = "data/matched_data"
    label_path = "data/AIS_Surgery_clean.csv"
    img_size = 512

    dataset = ScoliosisLumbarDataset(
        data_dir=dataset_dir,
        label_csv=label_path,
        img_size=img_size,
        mode="train",
        clinical_set="global_lumbar",
        use_cache=False,
        save_images=False,
        save_dir="lumbar/experiments_lumbar",
    )

    print("Clinical columns:", dataset.get_clinical_columns())
    print("Clinical dim:", dataset.get_clinical_dim())

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn
    )

    for i, (batch_data, labels) in enumerate(dataloader):
        print(f"Batch {i} loaded!")
        print("PA Shape:", batch_data["pa"].shape)
        print("LAT Shape:", batch_data["lat"].shape)
        print("Clinical Shape:", batch_data["clinical"].shape)
        print("Labels Shape:", labels.shape)
        print("Patient IDs:", batch_data["patient_id"])
        break
