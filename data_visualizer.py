import matplotlib.pyplot as plt

def visualizer(dataset, dataset_info, n: int):
    """
    iterate through the dataset object to unpack all the elements
    """
    for i, (image, label) in enumerate(dataset.take(n)):
        ax = plt.subplot(int(n**0.5), int(n**0.5), i+1)
        plt.imshow(image)
        plt.title(dataset_info.features['label'].int2str(label))
        plt.axis('off')
    plt.show()

def compare_dataset(original, augmented):
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    plt.imshow(augmented)

def visualize_cutmix(training_dataset):
    image, label = next(iter(training_dataset))
    plt.imshow(image[0])
