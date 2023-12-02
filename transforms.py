from torchvision import transforms

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomAffine(degrees=[-25,25], translate=[0.1,0.1]),
        transforms.Resize(232,antialias=False),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.CenterCrop(224)])

    test_transforms = transforms.Compose([
        transforms.Resize(232,antialias=False),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.CenterCrop(224)])
    return train_transforms, test_transforms
