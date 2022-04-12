from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, ToTensor
from model import VisionModule, DETR
from dataset import CustomDataset, load_data


if __name__ == '__main__':
    examples = load_data('/home/dewe/rgb_stacking_extend/rgb_stacking')
    sz = len(examples)
    train_sz, valid_sz = int(0.7 * sz), int(0.9 * sz)
    train_dt, valid_dt, test_dt = examples[:train_sz], examples[train_sz:valid_sz], examples[valid_sz:]

    img_transform = Normalize(0.1307, 0.3081)
    target_transform = ToTensor()

    train_ds, valid_ds, test_ds = CustomDataset(train_dt, img_transform, target_transform), \
                                  CustomDataset(valid_dt, img_transform, target_transform), \
                                  CustomDataset(test_dt, img_transform, target_transform)

    train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(valid_ds, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    model = VisionModule(7)

    print(model)

    print( model(train_features.float()) )