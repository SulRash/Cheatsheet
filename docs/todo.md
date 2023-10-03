# Experiments

## Controls

### Removing Appended Images

We should train with just the resized image without the appended images to the side, because that might have affected performance.

### Adding Noise

We should crop the original image randomly and assign it to different parts of the cheat sheet to introduce noise that isn't too far off from the original image.

# Unsupervised Learning

Train the model so that it doesn't use labelling and instead labels through comparing the embeddings of the main image and bounding boxes on the cheat sheet.

# Training Modifications

## Incorrect Labels and Shuffling the Cheatsheet

We should try shuffling the cheat sheet and changing the label of the image to the label of whatever replaced the image's position, this would force the model to look at the cheat sheet. How to do this and when should be experimented with.

# Datasets

## Imagenet

We should try and train with a subset of imagenet instead of CIFAR for more realistic performance.