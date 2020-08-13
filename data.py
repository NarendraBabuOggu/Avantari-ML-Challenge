"""
This file contains the dataset utils
"""
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import *
import warnings
warnings.filterwarnings('ignore')


def read_image(img_path: str, size: Optional[Union[int, Tuple[int, int]]] = None) -> Image:
    """
    Function to read the Image from a given path and resize it to a specific size

    params :
        img_path - string - contains the path of image
        size - Tuple - contains the height and width or integer indicating same height and width

    returns :
        A PIL.Image ytpe containing the required image
    """
    img = Image.open(img_path).convert("RGB")
    if size is not None:
        if isinstance(size, tuple) : 
            img = img.resize(size)
        elif isinstance(size, int) : 
            img = img.resize((size, size))
        else : 
            raise Exception(f"Invalid size argument for read_image : {size}")
    return img


class ImageDataset(Dataset):
    """
    A PyTorch Image Dataset with takes the paths of Images and returns the 
    Image Data that can be used for training the model
    """
    def __init__(self, image_paths: List[str], transform: Collection[Callable] = None,
                 size: Optional[Union[int, Tuple[int, int]]] = None ) :
        super(ImageDataset, self).__init__()
        self.image_paths = image_paths
        self.transform = transform
        self.size = size

    def __len__(self) -> int : 
        """
        Returns the Length of Dataset
        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, Any] : 
        """
        Returns the dict contating Image and its path for the given index
        """
        img_path = self.image_paths[index]
        img = read_image(img_path, self.size)
        img = self.transform(img)

        return {
            'image': img,
            'image_path': img_path
        }