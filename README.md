# Possible commands to be used in the project

## Commands for template.py
For the commands in template.py :: Each command starts with 'python template.py', to deter the repetetion we did not add this in front of each command below:

### custom model train and test :: custom layer_count kernel_size kernel_count learning_rate batch_norm tanh enable_test
E.g. "python template.py custom 2 5 16 0.07 false true true"
### run all models experimented in Baseline Arch. part :: base-all
E.g.  "python template.py base-all"
### run the best model experimented in Baseline Arch. part :: base-best
E.g.  "python template.py base-best"
### run all models experimented in Improved Arch. part :: improved-all
E.g.  "python template.py improved-all"
### run the best model experimented in Improved Arch. part :: improved-best
E.g.  "python template.py improved-best"

### run the best model obtained from the Improved Arch. part :: test-best-only
E.g. "python template.py test-best-only"

### run the additional model auto-encoder :: autoencoder enable_test
E.g. "python template.py autoencoder false"
### run the additional model unet :: unet enable_test
E.g. "python template.py unet false"
### run the additional model weirdnet :: weird enable_test
E.g. "python template.py weird false"


## Commands for namer.py :: directory_name
### Saving image file names into a txt files from choosen directory
E.g. "python namer.py train"


## Explanations and findings directory
resultant images
resultant loss and acc plots
resultant estimations_test.npy file 



## test_outputs directory
resultant images from 100 images
txt file containing the name of the images used 

## checkpoints directory
saved the best model for further uses