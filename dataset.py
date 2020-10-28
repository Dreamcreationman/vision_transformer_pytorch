import os
from util.data_process import DataProcess
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class ImagenetDataset(Dataset):

    def __init__(self, root_path, transform, train=True):
        self.root = root_path
        self.transform = transform
        self.train = train
        self.image_label_list = self.read_label(train)
        self.dataprocess = DataProcess(self.root)
        self.dataprocess.untar_dataset()
        self.toTensor = transforms.ToTensor()


    def __getitem__(self, item):
        image_path, image_label = self.image_label_list[item % len(self.image_label_list)]
        return image_path, image_label


    def __len__(self):
        return len(self.image_label_list)


    def read_label(self, train=True):
        image_label_list = []
        label_file_name = "train_label.txt" if train else "validation_label"
        label_file_path = os.path.join(self.root, label_file_name)
        if not os.path.exists(label_file_path):
            raise FileNotFoundError
        with open(label_file_path, "r", encoding="UTF-8") as f:
            lines = f.readlines()
            for line in lines:
                content = line.rstrip().split(' ')
                image_label_list.append((content[0], content[1]))
        return image_label_list

    def load_img(self, image_name, train):
        image = self.dataprocess.get_image(image_name, train)
        if self.transform is not None:
            return self.transform(image)
        else:
            return self.toTensor(image)
