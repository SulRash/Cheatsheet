from typing import Any, Callable, Optional, Tuple

import numpy as np
from torchvision.datasets.mnist import MNIST
from PIL import Image

class MNIST_Cheatsheet(MNIST):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        img_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        img_per_class: int = 0
    ) -> None:

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
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img, target = self.transform(img, original_target)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, original_target