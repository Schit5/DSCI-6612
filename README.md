# DSCI-6612
Project: Emotion AI 

Table of Contents: 

Introduction 

Overview 

Image Visualization 

Image Augmentation 

DATA NORMALIZATION AND TRAINING DATA PREPARATION 

DIVIDE DATA INTO TRAINING AND TESTING 

GRADIENT DESCENT 

Artificial neural networks 

Gradient descent 

Convolutional neural networks 

Resnets 

Training and testing data 

Deliverables 

Evaluation methodology 

 

 

Introduction 

Emotion AI refers to artificial intelligence that detects and interprets human emotional signals. The last thing a human expects is to replicate his own behaviour or expression and we are trying to build a model to predict one. The project focuses on 2 major points which are key point facial detection and facial expression prediction. 

 

Overview 

The aim of this project is to classify people’s emotions based on their face images. We have collected about 2000 images as  data and stored.  

Image Visualization 

Plottingrandom images from the dataset along with facial keypoints .Image data is obtained from df['Image'] and plotted using plt.imshow .Access their value using .loc command,where we get the values for  coordinates of the image based on the column it is refering to. 

 

Image Augmentation 

Obtain the columns in the dataframe .Flip the images along y axis .Y coordinate remains same as we are flipping horizontally .Only x coordiante values would change, all we have to do is to subtract our initial x-coordinate values from width of the image(96). Show Original image .Show Flipped image .Concatenate the original dataframe with the augmented dataframe .Increasing the brightness of the image. 

 

 

DATA NORMALIZATION AND TRAINING DATA PREPARATION 

 Normalize the images.Create an empty array of shape (x, 96, 96, 1) to feed the model.Iterate through the img list and add image values to the empty array after expanding it's dimension from (96, 96) to (96, 96, 1).Convert the array type to float32.Obtain the value of x & y coordinates which are to used as target. Split the data into train and test data 

 

DIVIDE DATA INTO TRAINING AND TESTING 

Data set is generally divided into 80% for training and 20% for testing.Sometimes, we might include cross validation dataset as well and then we divide it into 60%, 20%, 20% segments for training, validation, and testing, respectively (numbers may vary).Training set: Used for gradient calculation and weight update.Validation set: Used for cross -validation to assess training quality as training proceeds.Cross validation is implemented to overcome over-fitting which occurs when algorithm focuses on training set details at cost of losing generalization ability.Testing set: Used for testing trained network. 

 

 

 

 

GRADIENT DESCENT 

Gradient descent is an optimization algorithm used to obtain the optimized network weight and bias values.It works by iteratively trying to minimize the cost function.It works by calculating the gradient of the cost function and moving in the negative direction until the local/global minimum is achieved.If the positive of the gradient is taken, local/global maximum is achievedThe size of the steps taken are called the learning rate.If learning rate increases, the area covered in the search space will increase so we might reach global minimum faster. 

 

Artificial neural networks 

The brain has over 100 billion neurons communicating through electrical & chemical signals. Neurons communicate with each other and help us see, think, and generate ideas.The brain has over 100 billion neurons communicating through electrical & chemical signals. Neurons communicate with each other and help us see, think, and generate ideas.The neuron collects signals from input channels named dendrites, processes information in its nucleus, and then generates an output in a long thin branch called axon. 

 

Convolutional neural networks 

A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics. 

 

 

Resnets 

As CNNs grow deeper, vanishing gradient tend to occur which negatively impact network performance. Vanishing gradient problem occurs when the gradient is back -propagated to earlier layers which results in a very small gradient.Residual Neural Network includes “skip connection” feature which enables training of 152 layers without vanishing gradient issues.Resnet works by adding “identity mappings” on top of CNN.ImageNet contains 11 million images and 11,000 categories. ImageNet is used to train ResNet deep network. 

 

Deliverables 

A readme file.  

PY Code file.  

Evaluation methodology 

Data Normalization and Scaling  

ANNs Training & Gradient Descent Algorithm  

Build ResNet to Detect Key Facial Points  

Visualize Images for Facial Expression Detection  

Make Predictions from Both Models: 1. Key Facial Points & 2. Emotion  

Deploy Both Models and Make Inference  

From the Imported and Facial Expressions (Emotions) Datasets we compare our current image and the resulted prediction and thus compare the results for verification 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 
