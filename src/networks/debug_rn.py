import torch

from networks import resnet32, resnet32_no_bn, resnet32_ln

if __name__ == "__main__":
    resnet_standard = resnet32()
    resnet_no_bn = resnet32_no_bn()
    resnet_ln = resnet32_ln()

    input = torch.rand((1, 3, 32, 32))

    y1 = resnet_standard(input)
    y2 = resnet_no_bn(input)
    y3 = resnet_ln(input)

    print(y1.shape, y2.shape, y3.shape)
