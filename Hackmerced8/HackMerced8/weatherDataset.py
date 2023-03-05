import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class WeatherDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        date = row['DATE']
        min_temp = row['TMIN']
        max_temp = row['TMAX']
        precipitation = row['PRCP']
        return (date, torch.tensor([min_temp, max_temp, precipitation], dtype=torch.float))