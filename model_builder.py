import torch
from torch import nn
class TinyVGG(nn.Module):
    """TinyVGG.
    The model which is same as TinyVGG model

    Attributes:
        attribute: Description.
    """
    def __init__(self, input_shape :int ,
                 hidden_units : int ,
                 output_shape:int,
                 KERNEL_SIZE :int,
                 STRIDE:int,
                 PADDING:int):
        #self.input_shape = input_shape
        #self.hidden_units = hidden_units
        #self.output_shape = output_shape

        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE,
                      padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE,
                      padding=PADDING),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE,
                      padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE,
                      padding=PADDING),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)

        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 3* 3,
                      out_features=output_shape)
            
        )

    def forward(self,x:torch.Tensor):
        #print(f"Input tensor shape is {x.shape}")
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))
        
        
        
        
class VisionTransformed:
    """The class  replicates a vision transformer

    Attributes:
        attribute1 (type): Description of attribute1.
    """

    def __init__(self, params):
        """Initializes VisionTransformed .

        Args:
            params (type): Description of params.
        """
        
