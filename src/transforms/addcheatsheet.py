from math import ceil
from typing import Dict
from random import randint
from copy import copy

from PIL import Image

class AddCheatsheet():

    def __init__(self, sheet: Dict, cheatsheet: bool = False, cs_size: int = 8, num_classes: int = 10, cheatsheet_only: bool = False, randomize_sheet: bool = False) -> None:
        self.sheet = sheet
        self.cheatsheet = cheatsheet
        self.cs_size = cs_size
        self.num_classes = num_classes
        self.cheatsheet_only = cheatsheet_only
        self.randomize_sheet = randomize_sheet

    def __call__(self, img: Image, target: int):
        
        sheet = self.sheet
        max_images_in_row = 10
    
        new_image_box = self.cs_size * max_images_in_row
        additional_rows = self.cs_size * ceil(int(self.num_classes)/max_images_in_row)
        new_image_height = self.cs_size * max_images_in_row + additional_rows
        
        # Adds rows on the top for the cheatsheet
        if self.cheatsheet:

            if self.randomize_sheet:
                sheet, target = self.change_reference(target)

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

        if self.cheatsheet_only:
            if img in list(self.sheet.values()):
                return modified, target
            
        else:
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
    
        new_sheet = copy(self.sheet)

        replacement_id = randint(0, self.num_classes-1)
        replacement_image = self.sheet[replacement_id]
        target_image = self.sheet[target]

        new_sheet[target] = replacement_image
        new_sheet[replacement_id] = target_image

        target = replacement_id

        return new_sheet, target