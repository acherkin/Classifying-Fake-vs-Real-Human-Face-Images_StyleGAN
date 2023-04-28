# Classifying-Fake-vs-Real-Human-Face-Images_StyleGAN

Real VS Fake face classification
Anushka Bangal and Anna Cherkinsky
Group 3
Digital Image Processing

All code was created using python and Google Colab notebooks.
Please note that GPUs provided by Google Colab were used to run all code

The dataset was of Real human faces and fake StyleGAN-generated faces.
Link to the dataset: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
---------------------------------------------------------------------------------------
The code base contains the following files:


VGG_FakeVSReal.ipynb  --> Contains the initial VGG16 RGB binary classification attempt, the FFT VGG16 classification attept, the successful VGG16 + BatchNormalization layer attempt. It also contains model analysis for the VGG16 + 						BatchNormalization attempt, including the misclassified images, and feature map plots

Xception_initial_test.ipynb --> Contains initial FFT Xception classification attept, the RGB Xception attempt (initial)

Xception_Real_vs_Fake.ipynb --> Attempts to perform Xception classification with Batch Normalization

VGG Layer Norm and Inter-layer Batch Borm.ipynb --> Attempt to perform VGG16 classification with Layer Normalization instead of Batch Normalization. Also, an attempt to place BatchNormalization in the "in-between" layers of the VGG16 model.

VGG Unit Norm.ipynb --> Attempt to perform VGG16 classification with Unit Normalization instead of Batch Normalization. Also includes the run with BatchNormalizaition taken from VGG_FakeVSReal.ipynb, as a sanity check.

Data_FFT.ipynb --> Observations on a real and a fake image with FFT.

Metrics_Tests.ipynb --> code used to calculate the metrics for the models from "VGG Layer Norm and Inter-layer Batch Borm.ipynb", "Xception_Real_vs_Fake.ipynb", "VGG Unit Norm.ipynb"


---------------------------------------------------------------------------------------
Below are the pretrained model files:

Real VS Fake Xception.h5 --> Xception run from Xception_Real_vs_Fake.ipynb

Real VS Fake Xception2.h5 --> another Xception run from Xception_Real_vs_Fake.ipynb

Real VS Fake VGG_final2.h5 --> The complete Batch Normalization successful VGG run

Real VS Fake VGG_layer.h5 --> VGG16 with Layer normalization run from "VGG Layer Norm and Inter-layer Batch Borm.ipynb"

Real VS Fake VGG_unit.h5 --> VGG16 with Unit normalization run from "VGG Unit Norm.ipynb"

Real VS Fake Batch_normalization_mid.h5 --> VGG16 with in-between layers of batch normalization run from "VGG Layer Norm and Inter-layer Batch Borm.ipynb"


---------------------------------------------------------------------------------------
Below are all code sources and referenced used for this project


[10] Keras Team.  (n.d.). Keras Documentation: Xception. Keras. Retrieved April 25, 2023, from https://keras.io/api/applications/xception/ 
[11] Abdalrhmanmorsi. (2022, January 16). Real vs fake face CNN model. Kaggle. Retrieved April 25, 2023, from https://www.kaggle.com/code/abdalrhmanmorsi/real-vs-fake-face-cnn-model 
[12] Raulcsimpetru. (2019, July 4). VGG16 binary classification. Kaggle. Retrieved April 25, 2023, from https://www.kaggle.com/code/raulcsimpetru/vgg16-binary-classification 
[13] Easiest way to download kaggle data in Google Colab: Data Science and Machine Learning. Kaggle. (n.d.). Retrieved April 25, 2023, from https://www.kaggle.com/general/74235 
[14]FFT in python — python numerical methods - university of california ... (n.d.). Retrieved April 25, 2023, from https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.04-FFT-in-Python.html 
[15] Tf.keras.preprocessing.image.imagedatagenerator  :   tensorflow V2.12.0. TensorFlow. (n.d.). Retrieved April 25, 2023, from https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator 
[16] Xhlulu. (2020, February 10). 140k real and fake faces. Kaggle. Retrieved April 25, 2023, from https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces 
[17] Rosebrock, A. (2021, April 17). Fine-tuning with Keras and deep learning. PyImageSearch. Retrieved April 25, 2023, from https://pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/
[18] Hewage, R. (2020, May 15). Extract features, visualize filters and feature maps in VGG16 and VGG19 CNN Models. Medium. Retrieved April 28, 2023, from https://towardsdatascience.com/extract-features-visualize-filters-and-feature-maps-in-vgg16-and-vgg19-cnn-models-d2da6333edd0 
[19] Khandelwal, R. (2020, May 18). Convolutional Neural Network: Feature map and filter visualization. Medium. Retrieved April 28, 2023, from https://towardsdatascience.com/convolutional-neural-network-feature-map-and-filter-visualization-f75012a5a49c 
[20]Edeza, T. (2021, September 26). Image processing with python - application of Fourier transformation. Medium. Retrieved April 28, 2023, from https://towardsdatascience.com/image-processing-with-python-application-of-fourier-transformation-5a8584dc175b 
[21] Mcherukara. (n.d.). Mcherukara/FT_mnist: CNN Classifier on the Fourier transform of mnist data with &gt;95% accuracy. GitHub. Retrieved April 28, 2023, from https://github.com/mcherukara/FT_mnist
[22] Jimmy Lei Ba, Jamie Ryan Kiros, & Geoffrey E. Hinton. (2016). Layer Normalization. doi: 
https://doi.org/10.48550/arXiv.1607.06450
[23] Tf.keras.layers.layernormalization  :   tensorflow V2.12.0. TensorFlow. (n.d.). Retrieved April 28, 2023, from https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization 
[24] Tf.keras.layers.UnitNormalization  :   tensorflow V2.12.0. TensorFlow. (n.d.). Retrieved April 28, 2023, from https://www.tensorflow.org/api_docs/python/tf/keras/layers/UnitNormalization 
[25] Bhobé, M. (2019, July 14). Classifying fashion with a keras CNN (achieving 94% accuracy) - part 2. Medium. Retrieved April 28, 2023, from https://medium.com/@mjbhobe/classifying-fashion-with-a-keras-cnn-achieving-94-accuracy-part-2-a5bd7a4e7e5a
