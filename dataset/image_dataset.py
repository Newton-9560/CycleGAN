from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        with open(txt_file, 'r') as f:
            self.image_paths = f.read().splitlines()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image
    

def my_dataloader(txt_file, transformations, batch_size = 1, shuffle=True):
    dataset = ImageDataset(txt_file=txt_file, transform=transformations)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)