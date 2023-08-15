import os

import pandas as pd
import torch
from PIL import Image, ImageFile
from pytorch_lightning import seed_everything
from sklearn import preprocessing
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(42)
seed = seed_everything(42, workers=True)


class LokiDataset(Dataset):
    def __init__(self, img_transform=None, target_transform=None):
        self.df_abt = pd.read_csv(
            "output/update_allcruises_df_validated_5with_zoomie_20230303.csv"
        )
        self.df_abt = self.df_abt[self.df_abt["label"] != "Artefact"]  # remove artefacs
        self.label_encoder = preprocessing.LabelEncoder()
        self.image_root = self.df_abt["root_path"].values
        self.image_path = self.df_abt["img_file_name"].values
        self.label_encoder.fit(self.df_abt["label"])
        self.label = torch.Tensor(
            self.label_encoder.transform(self.df_abt["label"])
        ).type(torch.LongTensor)
        self.img_transform = img_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df_abt)

    def __getitem__(self, item):
        img_path = os.path.join(self.image_root[item], self.image_path[item])
        image = Image.open(img_path).convert("RGB")
        label = self.label[item]
        if self.img_transform:
            image = self.img_transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
