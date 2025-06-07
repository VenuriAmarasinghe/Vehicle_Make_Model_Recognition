# Vehicle_Make_Model_Recognition
Creating a model to recognize the model and make of vehicles 
first, the dataset can be downloaded from https://github.com/faezetta/VMMRdb?tab=readme-ov-file.

Next, the data set can be cleaned and split into train, val, and test in the ratio of 0.7:0.15:0.15 by running the data_clean_split.py file after giving the correct data path for the downloaded dataset.

Next, the training can be done by running the train.py file after specifying the paths of train and val data folders.

Next, the inferencing is done by running the inference.py 

converting to an ONNX file is shown in a jupyter notebook

calibration is done by the calibrate.py,
then we compare the original model, onnx model and the engine_int8 optimized model using compare.py
