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

## Masking the Reference Image

Masking the reference image corresponding to the label randomly during training might introduce some interesting results.

# Data

## Try Loading Cheatsheet at the Top Instead of Bottom

## Datasets

### Imagenet

We should try and train with a subset of imagenet instead of CIFAR for more realistic performance.

# Visualization

## Backpropogate to Input

We should investigate backpropogating the loss all the way to the input and draw heatmaps on the original image to see if changing the reference image's pixels would have affected model performance. This would show that the model is utilizing the reference image.

# Bugs

## Make Directory for Example Image if it Does Not Exist