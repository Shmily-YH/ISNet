import torch
from model.ISNet import ISNet

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = ISNet(num_classes=1,  output_stride=16, model_depth=50)
    model.eval()
    input = torch.randn((1, 3, 400, 400))
    output = model(input)
    print(output.shape)