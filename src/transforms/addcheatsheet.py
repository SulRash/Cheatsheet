from math import ceil
from typing import Dict

from PIL import Image

class AddCheatsheet():

    def __init__(self, sheet: Dict, cheatsheet: bool = False, cs_size: int = 8, num_classes: int = 10, cheatsheet_only: bool = False):
        self.sheet = sheet
        self.cheatsheet = cheatsheet
        self.cs_size = cs_size
        self.num_classes = num_classes
        self.cheatsheet_only = cheatsheet_only

    def __call__(self, img: Image, target: int):
        
        max_images_in_row = 10
    
        new_image_box = self.cs_size * max_images_in_row
        additional_rows = self.cs_size * ceil(int(self.num_classes)/max_images_in_row)
        new_image_height = self.cs_size * max_images_in_row + additional_rows
        
        # Adds rows on the top for the cheatsheet
        if self.cheatsheet:
            
            upscaled_image = img.resize((new_image_box, new_image_box))
            modified = Image.new('RGB', (new_image_box, new_image_height))
            modified.paste(upscaled_image, (0, additional_rows))
            
            image_rows = int(additional_rows/self.cs_size)
            for image_row in range(image_rows):
                
                remaining_images = min(len(self.sheet.keys()) - (max_images_in_row*image_row), max_images_in_row)
                for loc in range(remaining_images):
                    
                    # Set x and y axis locations to paste in cheatsheet image
                    x_loc = self.cs_size * loc
                    y_loc = self.cs_size * image_row

                    # Get number of cheatsheet image
                    cheatsheet_image = loc + (image_row*max_images_in_row)

                    modified.paste(self.sheet[cheatsheet_image].resize((self.cs_size, self.cs_size)), (x_loc, y_loc))
            
            left_over_black = max_images_in_row - remaining_images
            if left_over_black:
                x_loc = self.cs_size * (max_images_in_row - left_over_black)
                y_loc = self.cs_size * (image_rows - 1)
                modified.paste(Image.effect_noise((left_over_black*self.cs_size, self.cs_size), 25), (x_loc, y_loc))
                            
        else:
            modified = img.resize((new_image_box, new_image_height))

        if self.cheatsheet_only:
            if img in list(self.sheet.values()):
                return modified
            
        else:
            return modified