import os
import tarfile
from PIL import Image


class DataProcess:
    def __init__(self, root_path):
        self.root_path = root_path
        self.train_data_path = os.path.join(root_path, "ILSVRC2012_img_train")
        self.imagenet_test_path = os.path.join(self.root_path, "ILSVRC2012_img_test")
        self.imagenet_val_path = os.path.join(self.root_path, "ILSVRC2012_img_val")
        self.imagenet_train_path = os.path.join(root_path, "train")

    def un_tar(self, tar_file, output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        with tarfile.open(tar_file) as tar:
            names = tar.getnames()
            for name in names:
                tar.extract(name, output_path)

    def prepare_training_data(self):
        if os.path.exists(self.imagenet_train_path):
            print(" Training tar data has been untared!")
            return
        os.mkdir(self.imagenet_train_path)
        tar_images = [dir for dir in os.listdir(self.train_data_path) if os.path.isfile(os.path.join(self.train_data_path, dir)) and dir.endswith(".tar")]
        if len(tar_images) == 0:
            raise FileNotFoundError
        print("Start untar training tar data")
        for idx, tar in enumerate(tar_images):
            self.un_tar(os.path.join(self.train_data_path, tar), os.path.join(self.imagenet_train_path, tar.split(".")[0]))
            print("[{}/{}] has processed suceesfully!".format(idx+1, len(tar_images)))
        print("\nAll images have been extracted!")

    def untar_dataset(self):
        if not os.path.exists(self.train_data_path):
            raise FileNotFoundError
        else:
            self.prepare_training_data()
        if not os.path.exists(self.imagenet_test_path):
            os.mkdir(self.imagenet_test_path)
            self.un_tar(os.path.join(self.root_path, "ILSVRC2012_img_test.tar"), self.imagenet_test_path)
        if not os.path.exists(self.imagenet_val_path):
            os.mkdir(self.imagenet_test_path)
            self.un_tar(os.path.join(self.root_path, "ILSVRC2012_img_val.tar"), self.imagenet_val_path)


    def get_image(self, image_name, train=True):
        root = None
        if train:
            root = os.path.join(self.imagenet_train_path, image_name)
        else:
            root = os.path.join(self.imagenet_test_path, image_name)
        return Image.open(root)
