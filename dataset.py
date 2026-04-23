import torch
import pathlib
import PIL.Image
import pandas as pd
import sklearn.model_selection
import torchvision.transforms as transforms

def make_train_test_split(root, **kwargs):
    assert isinstance(root, pathlib.Path)

    paths, labels = (
        [str(path) for path in root.rglob("*/*.jpg")],
        [path.parent.stem for path in root.rglob("*/*.jpg")]
    )

    samples = pd.DataFrame(zip(paths, labels), columns=["path", "label"])

    samples_train, samples_test = sklearn.model_selection.train_test_split(
        samples,
        **kwargs,
        stratify=labels
    )

    return samples_train, samples_test

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None, target_transform=None):
        super().__init__()

        self.dataframe = dataframe
        self.transform = transform or (lambda sample: sample)
        self.target_transform = target_transform or (lambda sample: sample)

        self.class_to_idx = {
            category: index
            for (index, category) in enumerate(
                sorted(self.dataframe["label"].unique())
            )
        }

    def __getitem__(self, index):
        path, label = self.dataframe.iloc[index]
        image, label = (
            self.transform(PIL.Image.open(path).convert(mode="RGB")),
            self.target_transform(self.class_to_idx[label])
        )
        return image, label

    def __len__(self):
        return len(self.dataframe)

STD, MEAN = torch.tensor([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])

transform_extra = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=[0.08, 1], ratio=[3/4, 4/3]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandAugment(num_ops=2, magnitude=8),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD, inplace=True),
    transforms.RandomErasing(p=0.25, scale=[0.02, 0.2], ratio=[0.3, 3.3], inplace=True)
])

transform_basic = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=[256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD, inplace=True)
])

if __name__ == "__main__":
    dataframe_train, dataframe_test = make_train_test_split(pathlib.Path("/opt/datasets/imagenet-256-dimensi0n/"))

    dataset_train = ImageNetDataset(dataframe=dataframe_train, transform=transform_extra)
    dataset_test = ImageNetDataset(dataframe=dataframe_test, transform=transform_basic)

    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=64,
        shuffle=True,
        num_workers=8
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=128,
        shuffle=False,
        num_workers=8
    )

    print(dataset_train, dataset_test, dataloader_train, dataloader_test)
