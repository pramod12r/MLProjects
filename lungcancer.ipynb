!pip install timm scikit-learn matplotlib

import torch
import timm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import zipfile
import os
from tqdm import tqdm
import joblib

# Step 1: Upload and extract dataset (Expecting 'lung_data.zip' with cancer/normal folders)
from google.colab import files
uploaded = files.upload()

with zipfile.ZipFile("lung_data.zip", 'r') as zip_ref:
    zip_ref.extractall("lung_data")

print("Extracted folders:", os.listdir("lung_data"))

# Step 2: Load and transform images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder("lung_data", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Step 3: Load DINO ViT and extract CLS token features
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('vit_base_patch16_224_dino', pretrained=True)
model.eval().to(device)

all_features, all_labels = [], []
with torch.no_grad():
    for imgs, labels in tqdm(loader):
        imgs = imgs.to(device)
        features = model.forward_features(imgs)
        cls_features = features[:, 0, :]  # CLS token
        all_features.append(cls_features.cpu())
        all_labels.append(labels)

X = torch.cat(all_features).numpy()
y = torch.cat(all_labels).numpy()
print("Feature shape:", X.shape)

# Step 4: Train Random Forest Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Step 5: Evaluate
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred, target_names=dataset.classes))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
disp.plot(cmap='Blues')
plt.show()

# Step 6: Save model and download
joblib.dump(clf, "lung_rf_classifier.joblib")
files.download("lung_rf_classifier.joblib")
