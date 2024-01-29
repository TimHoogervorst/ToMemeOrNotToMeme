
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import timm
import matplotlib.pyplot as plt

# NN Classes / Datasets
class MemeOrCatDataSet(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes

# efficientnet_b0 pre-trained model
class efficientnet_b0(nn.Module):
    def __init__(self, num_classes):
        super(efficientnet_b0, self).__init__()

        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output


# The IDKWIWD Model (I Didn't know what I was doing model)
class MemeClassifier(nn.Module):
    def __init__(self, num_features=196608, h1=30, h2=20, features=2):
        super(MemeClassifier, self).__init__()
        self.fc1 = nn.Linear(num_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, features)

    def forward(self, x):
        # resolve this GPT fixed line?
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    

# First version of the confluence nn
class MemeClassConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convulence layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)

        # Pools
        self.pool = nn.MaxPool2d(2, 2)

        # Linear output
        self.fc1 = nn.Linear(32 * 28 * 28, 120) 
        self.fc2 = nn.Linear(120, 84) 
        self.out = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.shape[0])

        # Flatten for linear classification nodes
        x = x.view(-1, 32*28*28)

        # linear classification layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        
        return x

# Version two of Convulence layers
    # added Dropout layers in the classification section
    # Switched to grayscale images
    
class MemeClassConv2dV2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convulence layers
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.conv2 = nn.Conv2d(3, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)

        # Pools
        self.pool = nn.MaxPool2d(2, 2)

        # Linear output
        self.fc1 = nn.Linear(32 * 28 * 28, 120) 
        self.fc2 = nn.Linear(120, 84) 
        self.out = nn.Linear(84, 2)

        # DropOut Layer
        self.drop = nn.Dropout(0.2, inplace=False)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.shape[0])

        # Flatten for linear classification nodes
        x = x.view(-1, 32*28*28)

        # linear classification layers
        x = F.relu(self.fc1(x))
        x = self.drop(x)

        x = F.relu(self.fc2(x))
        x = self.drop(x)

        x = self.out(x)
        return x

# Functions for Predicting images
    
def predict(model, image_tensor, device, class_names):
    tensor = image_tensor.to(device)
    output = model.forward(tensor).argmax().cpu().item()
    return class_names[output]

def predict_softmax(model, image_tensor, device):
    # Predict using the model
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

def visualize_predictions(original_image, probabilities, class_names):
    # Visualization
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    # Display image
    ax[0].imshow(original_image)
    ax[0].axis("off")
  
    # Display predictions
    ax[1].barh(class_names, probabilities)
    ax[1].set_xlabel("Probability")
    ax[1].set_title("Class Predictions")
    ax[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()

def main():
    pass

if __name__ == "__main__":
    main()

