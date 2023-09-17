from sklearn.calibration import LabelEncoder
from torchvision import datasets, transforms
from data.manipulate import UnNormalize
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import math

LABEL_COLUMN = 'Attack Type'
NUM_COLUMNS = 81
IMAGE_EDGE_SIZE = int(math.sqrt(NUM_COLUMNS))
NUM_CLASSES = 9

def setup_dataset(df: pd.DataFrame, verbose=True):
    shrink_dataset_for_demo(df, 5000)

    verbose and print('\nClasses percentage:', df[LABEL_COLUMN].value_counts(normalize=True))
    verbose and print('\nClasses samples count:', df[LABEL_COLUMN].value_counts(normalize=False))

    for i in range(NUM_COLUMNS - len(df.columns) + 1):
        df[f'nan{i}'] = 0

    print(f'\nNumber of columns {len(df.columns)}')

def shrink_dataset_for_demo(df, x_to_leave_from_each_class):
    class_values = df[LABEL_COLUMN].unique()
    for cls in class_values:
        l = len(df.index[df[LABEL_COLUMN] == cls])
        p = 1 - min(l, x_to_leave_from_each_class) / l
        reduce_class(df, LABEL_COLUMN, cls, p)

def reduce_class(df, col_name, class_val, p):
    indices = df.index[df[col_name] == class_val]
    df.drop(indices[:int(len(indices) * p)], inplace=True)

def split_dataset(dataframe):
    y_data = dataframe[LABEL_COLUMN].values
    x_data = dataframe.drop(columns=[LABEL_COLUMN]).values
    return x_data, y_data


class networkDataset(Dataset):

    @property
    def train_labels(self):
        return self.targets
    
    @property
    def train_data(self):
        return self.data

    def __init__(self, src_file, train=True, none=False, all=False, verbose=False, transform=None, target_transform=None, download=False, transforms=None):
        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can be passed as argument")
        
        if has_separate_transform:
            transforms = datasets.vision.StandardTransform(transform, target_transform)
        self.transforms = transforms
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if none:
            self.data, self.targets = None, None
            return

        df = pd.read_csv(src_file)
        setup_dataset(df)
        le = LabelEncoder()
        df[LABEL_COLUMN] = le.fit_transform(df[LABEL_COLUMN])
        if verbose:
            print("\nLabelEncoder mappings:")
            for i, class_label in enumerate(le.classes_):
                print("{0} --> {1}".format(class_label, i))
        x_data, y_data = split_dataset(df)

        if all:
            self.data, self.targets = x_data, y_data
            return
        
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

        if train:
            self.data, self.targets = x_train, y_train
        else:
            self.data, self.targets = x_test, y_test
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        vector, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.reshape(vector, (IMAGE_EDGE_SIZE,IMAGE_EDGE_SIZE)), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


# specify available data-sets.
AVAILABLE_DATASETS = {
    'MNIST': datasets.MNIST,
    'CIFAR100': datasets.CIFAR100,
    'CIFAR10': datasets.CIFAR10,
    '5GNIDD': networkDataset,
}

# specify available transforms.
AVAILABLE_TRANSFORMS = {
    'MNIST': [
        transforms.ToTensor(),
    ],
    'MNIST32': [
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    'CIFAR10': [
        transforms.ToTensor(),
    ],
    'CIFAR100': [
        transforms.ToTensor(),
    ],
    '5GNIDD': [
        transforms.ToTensor(),
    ],
    'CIFAR10_norm': [
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ],
    'CIFAR100_norm': [
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
    ],
    'CIFAR10_denorm': UnNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    'CIFAR100_denorm': UnNormalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
    'augment_from_tensor': [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ],
    'augment': [
        transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
    ],
}

# specify configurations of available data-sets.
DATASET_CONFIGS = {
    'MNIST': {'size': 28, 'channels': 1, 'classes': 10},
    'MNIST32': {'size': 32, 'channels': 1, 'classes': 10},
    'CIFAR10': {'size': 32, 'channels': 3, 'classes': 10},
    'CIFAR100': {'size': 32, 'channels': 3, 'classes': 100},
    '5GNIDD': {'size':IMAGE_EDGE_SIZE, 'channels': 1, 'classes': NUM_CLASSES},
}
