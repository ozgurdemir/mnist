# Basic CNN model to predict Handwritten digits

This is a simple Keras CNN model to predict handwritten digits based on the Mnist data set.

This repository might be used to experiment with different model structures, data normalizations, 
weight initializations etc.

## How to use

To create the docker image run:

        > make build
        
To train the model and save the model weights to `data` folder:

        > make train
        
The above step will also run a validation on a validation set and report the performance.

Additional arguments may be passed via

        > make train --model PATH/TO/MODEL/WEITHS ...