import timm

from models.resnet import ResNet18, ResNet34
import timm

def get_model(model_name: str, num_classes: int = 10, cs_size: int = 8):
    
    if model_name == "resnet18":
        return ResNet18(num_classes)
    elif model_name == "resnet34":
        return ResNet34(num_classes)
    else:
        try:
            return timm.create_model(
                model_name, num_classes=num_classes, img_size=cs_size*10
            )
        except:
            print("Invalid model!")
            exit()