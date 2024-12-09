### Image Captioning

Download the dataset used: https://www.kaggle.com/dataset/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb
Then set images folder, captions.txt inside a folder flickr8k.

Download my pre-trained weights:
https://www.kaggle.com/datasets/rajeshviswa1/imagecaptioning-weights

To train the model:
Set num_echoes in train.py to a desired value.
Run train.py

To test the pretrained model:
Run eval.py

train.py: For training the network

model.py: creating the encoderCNN, decoderRNN and hooking them together

get_loader.py: Loading the data, creating vocabulary

utils.py: Load model, save model, printing test cases downloaded online

eval.py: To print some test cases over a new sample of images

