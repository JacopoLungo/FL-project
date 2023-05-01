import os
from typing import Any
import numpy as np
from PIL import Image
from torch import from_numpy
from torchvision.datasets import VisionDataset
import datasets.ss_transforms as tr
import matplotlib.pyplot as plt

class_eval = [255, 2, 4, 255, 11, 5, 0, 0, 1, 8, 13, 3, 7, 6, 255, 255, 15, 14, 12, 9, 10]
trasformato = [255, 0, 1, 255, 2, 3, 255, 4, 5, 6,  7,  8,  9, 10, 255, 255,  11, 12, 13, 14, 15, 255, 255, 255]


class IDDADataset(VisionDataset):

    def __init__(self,
                 root: str,
                 list_samples: [str],
                 transform: tr.Compose = None,
                 client_name: str = None):
        super().__init__(root=root, transform=transform, target_transform=None)
        self.list_samples = list_samples
        self.client_name = client_name
        self.target_transform = self.get_mapping()
        self.root = root

    @staticmethod
    def get_mapping(mappa = class_eval):
        """
        mappa le labels che non usiamo a 255 e scala le altre.
        Esempio:
        mappa = [255, 0, 1, 2]
        originale = [3, 2,   0, 2, 1]
        mappato =   [2, 1, 255, 1, 0]
        """
        classes = mappa
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for i, cl in enumerate(classes):
            mapping[i] = cl
        return lambda x: from_numpy(mapping[x])

    def __getitem__(self, index: int) -> Any:
        """
        :return: torch.tensor()
        """
        # TODO: missing code here!
        imagePath = os.path.join(self.root, 'images' , self.list_samples[index]+'.jpg')
        labelPath = os.path.join(self.root, 'labels' ,self.list_samples[index]+'.png')

        image = Image.open(imagePath)
        label = Image.open(labelPath)

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, IDDADataset.get_mapping(trasformato)(label)

    def __len__(self) -> int:
        return len(self.list_samples)
        
    def unNormalize(self, tensorImage, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        for t, m, s in zip(tensorImage, mean, std):
            t.mul_(s).add_(m)
        return tensorImage
    #TODO: chimare unNormalized da utils e toglierlo da qui 
    def showImgAndLable(self, index):
        image, label = self.__getitem__(index)
        #unnormalize to show
        fig, axs = plt.subplots(2)
        #fig.set_figheight(8.275)
        #fig.set_figwidth(15)
        fig.suptitle('Transformed and Segmentated Image')
        #fig.tight_layout()
        for i in range(2):
            axs[i].set_xticks([])
            axs[i].set_yticks([])

        axs[0].imshow(self.unNormalize(image).permute(1, 2, 0))
        axs[1].imshow(label.numpy())
