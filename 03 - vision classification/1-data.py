from data import collections, data_loader


def example(collection, type):
    dataloader = data_loader(collection, type)
    images, labels = next(iter(dataloader))
    image = images[0]
    label = labels[0]
    print(f"Data: {collection}/{type} [{len(dataloader.dataset)}]")
    print(f"Classes: {len(dataloader.dataset.classes)}")
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}, {dataloader.dataset.classes[label]}")
    print()


print()
for collection in collections:
    example(collection, "train")
    example(collection, "test")
