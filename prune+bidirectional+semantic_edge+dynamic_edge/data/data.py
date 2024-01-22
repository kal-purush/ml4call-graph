"""
    File to load dataset based on user control from main file
"""
from data.CustomDatasetWithDroppingNodes import CustomDatasetWithDroppingNodes
from data.DatasetWithCustomNode import DatasetWithCustomNode
from data.DatasetWithFunctionAndChildNode import DatasetWithFunctionAndChildNode
from data.DatasetWithFunctionEdge import DatasetWithFunctionEdge
from data.DatasetWithSelectedNode import DatasetWithSelectedNode
from data.NewDataset import NewDataset
from data.cycles import CyclesDataset




def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """
    # handling for MNIST or CIFAR Superpixels
    
    if 'NEW_DATASET' in DATASET_NAME:
        parts = DATASET_NAME.split("--")
        APPLICATION_NAME = parts[1]
        return NewDataset(APPLICATION_NAME)
    