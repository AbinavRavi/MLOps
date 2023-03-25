from torch.utils.data import random_split,  DataLoader, Dataset
import pytorch_lightning as pl
from typing import Optional
import numpy as np
import os
import polars as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re

class TextPreprocess:
    def __init__(self,text) -> None:
        self.text = text

    def convert2lower():
        pass

    def tokenization():
        pass

    def remove_stop_words():
        pass

    def stemming():
        pass

    def lemmatization():
        pass

    def remove_special_chars():
        pass
    
class CleanTweets(Dataset):
    """ 
    Allows to read the input and target data, 
    combine them to a pair of tensors which then can be passed to specific dataloaders 
    and finally to the train / validate or test the model 
    Args:
        Dataset (object)
        
    """
    def __init__(self, root_path):
        """
        NEEDED: Constructor for dataset class for semantic segmentation.
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        
        Args:
            root_path (str): path to data
            ipt (str): specific path to input data 
            tgt (str): specific path to target (mask) data
            tgt_scale (int): defining the target map scale
            train_transform (bool, optional): Applies the defined data transformations used in training. Defaults to None.
        """
        super(CleanTweets, self).__init__()
        self.root_path = root_path
        df = pd.read_csv(self.root_path)
        self.text = df["tweet"]
        self.label = df["target"]

    
    def __len__(self):
        """
        Necessary function returning the length of the dataset
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        
        Returns:
            _type_: _description_
        """
        l1 = os.listdir(os.path.join(self.root_path, self.ipt)) # dir is your directory path
        l2 = os.listdir(os.path.join(self.root_path, self.tgt)) # dir is your directory path
        number_files_inp = len(l1)
        number_files_tgt = len(l2)

        if number_files_inp == number_files_tgt:
            return number_files_inp

    def __getitem__(self, idx):
        """
        Necessary function that loads and returns a sample from the dataset at a given index. 
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        
        Based on the index,it identifies the input and target images location on the disk, 
        reads both items as a numpy array (float32). If the train_transform argument is True, 
        the above defined train transformations are applied. Else, the test transformations are applied
        Args:
            idx (iterable): 
        Returns:
            tensors: input and target image
        """
        # change here so that in the class the scale can be defined not hard coded 'geb_' or 'geb_10'
        img_path_input_patch = os.path.join(self.root_path, self.ipt, f"geb_{idx}.npy")
        img_path_tgt_patch = os.path.join(self.root_path, self.tgt, f"geb_{str(self.tgt_scale)}_{idx}.npy")
        
        input_patch = np.load(img_path_input_patch).astype('float32')
        tgt_patch = np.load(img_path_tgt_patch).astype('float32')
            
        if self.train_transform:
            input_patch, tgt_patch = self.my_train_segmentation_transforms(input_patch, tgt_patch)
        
        else: 
            input_patch, tgt_patch = self.my_test_segmentation_transforms(input_patch, tgt_patch)
            
        return input_patch, tgt_patch        
        
class MyDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self,):
        super(MyDataModule).__init__()
        
    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally. 
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """
      
        # Assign train/val datasets for use in dataloaders
        # the stage is used in the Pytorch Lightning trainer method, which you can call as fit (training, evaluation) or test, also you can use it for predict, not implemented here
        
        if stage == "fit" or stage is None:
            train_set_full =  CleanTweets(
                root_path="/Users/yourusername/path/to/data/train_set/",
                ipt="input/",
                tgt="target/",
                tgt_scale=25, 
                train_transform=True)
            train_set_size = int(len(train_set_full) * 0.9)
            valid_set_size = len(train_set_full) - train_set_size
            self.train, self.validate = random_split(train_set_full, [train_set_size, valid_set_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = CleanTweets(
                root_path="/Users/yourusername/path/to/data/test_set/",
                ipt="input/",
                tgt="target/",
                tgt_scale=25, 
                train_transform=False)
            
    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet. 
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=32, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32, num_workers=8)