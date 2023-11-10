from typing import Any, Callable, Optional, Tuple

import numpy as np
from torchvision.datasets.cifar import CIFAR10
from PIL import Image

class CIFAR_Cheatsheet(CIFAR10):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        img_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        dataset_name: str = "cifar10",
        img_per_class: int = 0
    ) -> None:

        if dataset_name == "cifar10":
            self.base_folder = "cifar-10-batches-py"
            self.url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            self.filename = "cifar-10-python.tar.gz"
            self.tgz_md5 = "c58f30108f718f92721af3b95e74349a"
            self.train_list = [
                ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
                ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
                ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
                ["data_batch_4", "634d18415352ddfa80567beed471001a"],
                ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
            ]

            self.test_list = [
                ["test_batch", "40351d587109b95175f43aff81a1287e"],
            ]
            
            self.meta = {
                "filename": "batches.meta",
                "key": "label_names",
                "md5": "5ff9c542aee3614f3951f8cda6e48888",
            }
        
        elif dataset_name == "cifar100":
            self.base_folder = "cifar-100-python"
            self.url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
            self.filename = "cifar-100-python.tar.gz"
            self.tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
            self.train_list = [
                ["train", "16019d7e3df5f24257cddd939b257f8d"],
            ]

            self.test_list = [
                ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
            ]
            
            self.meta = {
                "filename": "meta",
                "key": "fine_label_names",
                "md5": "7973b15100ade9c7d40fb424638fde48",
            }
        super().__init__(root, train, transform, target_transform, download)
        
        original_size = self.__len__()

        self.img_transform = img_transform

        # Grabbing desired number of images per class
        new_indices = []
        if img_per_class:
            
            for idx in range(10):
                class_indices = [i for i, x in enumerate(self.targets) if x == idx]
                class_indices_sliced = class_indices[:img_per_class]
                new_indices.extend(class_indices_sliced)

            # Increasing dataset size back up by duplicating
            self.data, self.targets = self.data[new_indices], np.array(self.targets)[new_indices]
            self.data, self.targets = np.resize(self.data, (original_size,32,32,3)), np.resize(self.targets, original_size)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, original_target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img, target = self.transform(img, original_target)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.transform is not None:
            return img, target, original_target
        
        return img, original_target
