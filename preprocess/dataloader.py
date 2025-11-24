from torch.utils.data import DataLoader, WeightedRandomSampler, default_collate
from .dataset import FaceDataset
from .transforms import get_transforms

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)

def build_loaders(
    train_csv="labeled/train/label.csv", train_img="labeled/train/img",
    val_csv="labeled/valid/label.csv",     val_img="labeled/valid/img",
    test_csv="labeled/test/label.csv",   test_img="labeled/test/img",
    img_size=224, batch_size=32, num_workers=0, pin_memory=False,
    use_weighted_sampler=False, validate_files=False, drop_last=False
):
    train_t, val_t, test_t = get_transforms(img_size)

    train_ds = FaceDataset(train_csv, train_img, transform=train_t, validate_files=validate_files)
    val_ds   = FaceDataset(val_csv,   val_img,   transform=val_t,   validate_files=validate_files)
    test_ds  = FaceDataset(test_csv,  test_img,  transform=test_t,  validate_files=validate_files)

    sampler = None
    if use_weighted_sampler:
        g = train_ds.df["gender"].astype(int)
        counts = g.value_counts().to_dict()
        weights = {c: 1.0 / counts[c] for c in counts}
        sample_weights = g.map(weights).values
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_ld = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=drop_last, collate_fn=safe_collate
    )
    val_ld = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=False, collate_fn=safe_collate
    )
    test_ld = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=False, collate_fn=safe_collate
    )
    return train_ld, val_ld, test_ld
