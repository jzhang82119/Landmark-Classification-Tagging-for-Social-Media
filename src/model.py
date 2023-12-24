import torch
import torch.nn as nn


# class MyModel(nn.Module):
#     def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
#         super().__init__()
#         # YOUR CODE HERE
#         # Define a CNN architecture. Remember to use the variable num_classes
#         # to size appropriately the output of your classifier, and if you use
#         # the Dropout layer, use the variable "dropout" to indicate how much
#         # to use (like nn.Dropout(p=dropout))
#         self.conv1 = nn.Sequential(
#           nn.Conv2d(3, 16, 3, padding=1),
#           nn.MaxPool2d(2,2),
#           nn.ReLU()                # 112 x 112
#         )
        
#         self.conv2 = nn.Sequential(
#           nn.Conv2d(16, 32, 3, padding=1),
#           nn.BatchNorm2d(32),
#           nn.MaxPool2d(2,2),
#           nn.ReLU(),
#           nn.Dropout2d(p=dropout)   # 56 x 56
#         )
        
#         self.conv3 = nn.Sequential(
#           nn.Conv2d(32, 64, 3, padding=1),
#           nn.BatchNorm2d(64),
#           nn.MaxPool2d(2,2),
#           nn.ReLU(),
#           nn.Dropout2d(p=dropout)      # 28x28
#         )
        
#         self.conv4 = nn.Sequential(
#           nn.Conv2d(64, 128, 3, padding=1),
#           nn.BatchNorm2d(128),
#           nn.MaxPool2d(2,2),
#           nn.ReLU(),
#           nn.Dropout2d(p=dropout)         # 14x14
#         )
        
#         self.conv5 = nn.Sequential(
#           nn.Conv2d(128, 256, 3, padding=1),
#           nn.BatchNorm2d(256),
#           nn.MaxPool2d(2,2),
#           nn.ReLU(),
#           nn.Dropout2d(p=dropout)         # 7x7
#         )
        
        
#         self.fc = nn.Sequential(
#             nn.Linear(7*7*256, 4096),
#             nn.BatchNorm1d(4096),
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
            
#             nn.Linear(4096, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
            
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
            
#             nn.Linear(512, num_classes)
#         )
        
        
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # YOUR CODE HERE: process the input tensor through the
#         # feature extractor, the pooling and the final linear
#         # layers (if appropriate for the architecture chosen)
#         x=self.conv1(x)
#         x=self.conv2(x)
#         x=self.conv3(x)
#         x=self.conv4(x)
#         x=self.conv5(x)
#         x=torch.flatten(x, 1)
#         x=self.fc(x)
#         return x



class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),        # 112 x 112
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),        # 56 x 56
            
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),        # 28x28
            
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),        # 14x14
            
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),        # 7x7
            
            
            nn.Flatten(),
            
            nn.Linear(7*7*256, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            
            nn.Linear(512, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)

######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
