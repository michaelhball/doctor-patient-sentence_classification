This program is run entirely from main.py, using command line arguments. To see the list of options, you can run python main.py -h.

But in general, the key arguments are --encoder_type, --classifier_type, and --task (which outlines what the program will do), as the defaults for the rest should be suitable. Of course if you want to train a model for the extended classifier problem then that'll have to be set by the --classfication_type argument etc. etc. The rest are explained using the help command. 

Beyond that, the only real --task options that need to be considered are 'train' and 'model_accuracy.' The former trains a single model using its best collection of hyperparameters, visualises the training process, and returns the final test classification accuracy of that model. Model_accuracy creates 10 classifiers according to the same specifications, and averages the final test classification accuracies in order to get a solid value for a given classifier parametrisation.

The 'output_csv' task is what I used to make the classifications and write to a CSV for all test examples. It currently looks for a trained model stored in the saved_models subdirectory, loads those weights, and then moves through the testing examples classifying both simple and extended cases.

--word_embedding options for the pooling classifier are fasttext_300 and glove_50.

I have included all data objects in the zip file, as these are used to speed up training and experimenting. In particular, the test_data... and train_data... files consist of exactly the same data as the given tsvs, but with words stored as their word embeddings. This is because the initial embedding process is slow, so I use these files for training when models use word embeddings as part of the input.

In addition, the vocab (indexing words for use in bag of words), and label indexes for simple and extended are all stored as pickle objects. As long as this directory structure is maintained, you shouldn't have to worry about any of these as they'll get used automatically in the main method of main.py according to the input arguments.
