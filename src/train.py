import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models


# -------------------------
# Dataset
# -------------------------
class CellDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        classes = {
            "cell": 1,
            "nocell": 0
        }

        for class_name in classes:
            class_path = os.path.join(root_dir, class_name)

            for img_name in os.listdir(class_path):
                if img_name.endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(class_path, img_name)

                    self.samples.append(
                        (img_path, classes[class_name])
                    )

        print(f"Loaded {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# -------------------------
# Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


# -------------------------
# Dataset Loader
# -------------------------
dataset = CellDataset(
    root_dir="data/images",
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)


# -------------------------
# Model
# -------------------------
model = models.resnet18(weights="DEFAULT")

model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)


# -------------------------
# Training
# -------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 5

for epoch in range(epochs):

    running_loss = 0.0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss: {running_loss:.4f}"
    )


# -------------------------
# Save Model
# -------------------------
os.makedirs("models", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/cell_model.pth"
)

print("Model saved!")
