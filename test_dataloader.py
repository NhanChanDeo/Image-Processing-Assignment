import torch
from preprocess.dataloader import build_loaders

def main():
    train_ld, val_ld, test_ld = build_loaders(
        batch_size=8, num_workers=0, img_size=224,
        use_weighted_sampler=False, validate_files=False
    )

    batch = next(iter(train_ld))
    while batch is None:
        batch = next(iter(train_ld))

    imgs, targets = batch
    assert isinstance(imgs, torch.Tensor) and imgs.shape == (8, 3, 224, 224)
    assert isinstance(targets, dict) and "age" in targets and "gender" in targets
    assert targets["age"].shape[0] == 8 and targets["gender"].shape[0] == 8
    assert int(targets["gender"].min()) >= 0 and int(targets["gender"].max()) <= 1

    print("batch:", tuple(imgs.shape))
    print("age:", [float(a) for a in targets["age"][:3]])
    print("gender:", [int(g) for g in targets["gender"][:3]])

if __name__ == "__main__":
    main()
