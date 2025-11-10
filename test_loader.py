import torch
import matplotlib.pyplot as plt
import numpy as np

from preprocess import FaceDataset, get_transforms

train_t, val_t, _ = get_transforms(224)
train_ds = FaceDataset("labeled/train/label.csv", "labeled/train/img", transform=train_t)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)

def denorm(img_tensor):
    x = img_tensor.numpy()           
    x = x * IMAGENET_STD + IMAGENET_MEAN
    x = np.clip(x, 0, 1)
    return x.transpose(1, 2, 0)      

imgs, targets = next(iter(train_ld))  
ages = targets["age"].numpy().astype(int)
genders = targets["gender"].numpy()

plt.figure(figsize=(12, 6))
for i in range(len(imgs)):
    plt.subplot(2, 4, i+1)
    plt.imshow(denorm(imgs[i].cpu()))
    gtxt = "Nam" if genders[i]==0 else "Nữ"
    plt.title(f"Age: {ages[i]}, {gtxt}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("test_loader.png", dpi=150, bbox_inches="tight")
plt.show()

