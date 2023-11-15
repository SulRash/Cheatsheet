from math import ceil
from typing import Dict
from random import randint, shuffle

from PIL import Image

class AddCheatsheet():

    def __init__(self, sheet: Dict, num_classes: int, args) -> None:
        self.sheet = sheet
        self.num_classes = num_classes

        self.cheatsheet = args.cheatsheet
        self.cs_size = args.cs_size
        self.randomize_sheet = args.randomize_sheet
        self.one_image = args.one_image
        self.one_image_per_class = args.one_image_per_class
        self.concat = args.concat

    def __call__(self, img: Image, target: int):

        sheet = self.sheet

        if self.one_image:
            img = sheet[0]
            target = 0
        elif self.one_image_per_class:
            img = sheet[target]
        
        # Adds rows on the top for the cheatsheet
        if self.cheatsheet:

            if self.randomize_sheet:
                sheet, target = self.change_reference(target)

            if self.concat:
                new_image_width = img.size[0] * (self.num_classes+1)
                modified = modified = Image.new('RGB', (new_image_width, img.size[1]))
                modified.paste(img, (0,0))
                for i in range(self.num_classes):
                    x_loc = img.size[0] * (i+1)
                    modified.paste(sheet[i], (x_loc,0))

            else:

                max_images_in_row = 10
                new_image_box = self.cs_size * max_images_in_row
                additional_rows = self.cs_size * ceil(int(self.num_classes)/max_images_in_row)
                new_image_height = self.cs_size * max_images_in_row + additional_rows

                modified = self.add_cheatsheet_rows(
                    img=img,
                    sheet=sheet,
                    max_images_in_row=max_images_in_row,
                    new_image_box=new_image_box,
                    new_image_height=new_image_height,
                    additional_rows=additional_rows
                )
        else:
            modified = img.resize((new_image_box, new_image_height))
            
        
        return modified, target
    
    def add_cheatsheet_rows(self, img, sheet, max_images_in_row, new_image_box, new_image_height, additional_rows):

        upscaled_image = img.resize((new_image_box, new_image_box))
        modified = Image.new('RGB', (new_image_box, new_image_height))
        modified.paste(upscaled_image, (0, additional_rows))
        
        image_rows = int(additional_rows/self.cs_size)
        for image_row in range(image_rows):
            
            remaining_images = min(self.num_classes - (max_images_in_row*image_row), max_images_in_row)
            for loc in range(remaining_images):
                
                # Set x and y axis locations to paste in cheatsheet image
                x_loc = self.cs_size * loc
                y_loc = self.cs_size * image_row

                # Get number of cheatsheet image
                cheatsheet_image = loc + (image_row*max_images_in_row)

                modified.paste(sheet[cheatsheet_image].resize((self.cs_size, self.cs_size)), (x_loc, y_loc))
        
        left_over_black = max_images_in_row - remaining_images
        if left_over_black:
            x_loc = self.cs_size * (max_images_in_row - left_over_black)
            y_loc = self.cs_size * (image_rows - 1)
            modified.paste(Image.effect_noise((left_over_black*self.cs_size, self.cs_size), 25), (x_loc, y_loc))
        
        return modified

    def change_reference(self, target):
        new_sheet = {}
        shuffled_labels = list(self.sheet.keys())
        shuffle(shuffled_labels)
        target = shuffled_labels.index(target)
        for idx, label in enumerate(shuffled_labels):
            new_sheet[idx] = self.sheet[label]

        return new_sheet, target