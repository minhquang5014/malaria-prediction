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

