from data_prep import sample_data, splits

TRAIN_RATIO = 0.9
VAL = 0.05
TEST = 0.05

# in here, we got the dataset and information of the dataset
dataset, dataset_info = sample_data()

# print(sample_data())

# split the data into train dataset, validation data, and test set
train_dataset, val_dataset, test_dataset = splits(dataset[0], TRAIN_RATIO, VAL, TEST)

print(len(train_dataset), len(val_dataset), len(test_dataset))