import torchvision.transforms as transforms

class ToBfloat16(transforms.ToTensor):

    def __init__(self):
        super().__init__()

    def __call__(self, img):
        return img.bfloat16()
    