from math import ceil

import timm

from models.resnet import ResNet18, ResNet34

def get_model(model_name: str, num_classes: int = 10, cs_size: int = 8, concat: bool = False):

    max_images_in_row = 10
    if concat:
        new_image_box = 32*11
        new_image_height = 32
    else:
        new_image_box = cs_size * max_images_in_row
        additional_rows = cs_size * ceil(int(num_classes)/max_images_in_row)
        new_image_height = cs_size * max_images_in_row + additional_rows
    
    if "resnet" in model_name:
        return timm.create_model(
            model_name, num_classes=num_classes, pretrained=False
        )
    else:
        try:
            return timm.create_model(
                    model_name, num_classes=num_classes, img_size=(new_image_height, new_image_box), pretrained=False
                )
        except:
            print("Invalid model!")
            exit()