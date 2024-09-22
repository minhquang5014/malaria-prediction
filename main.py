from data_prep import sample_data, splits
from data_visualizer import visualizer

# in here, we got the dataset and information of the dataset
dataset, dataset_info = sample_data()

# split the data into train dataset, validation data, and test set
train_dataset, val_dataset, test_dataset = splits(dataset[0])

visualizer(val_dataset, dataset_info, 25)
