#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output-images/first-turn.jpg "First turn"
[image2]: ./output-images/backwards.jpg "Backward through track"
[image3]: ./output-images/problem-area.jpg "Recovery Image"
[image4]: ./output-images/cropped-image.png "Cropped Image"


## Rubric Points

###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* output-run.mp4

Here's a [link to my video result](./output-images/output-run.mp4)

####2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based on the model architecture developed by NVidia in their paper "Explaing How a Deep Neural Network Trained 
with End-to-End Learning Steers a Car". I followed the guidance outlined in the paper, with the exception of adding
a dropout layer after the convolutional layers.  

The model is implemented using the Keras framework.  The code snippet below shows the implementation  
found in `model.py` at line 29 

```
   # Crop image to critical area.
    model.add(Cropping2D(cropping=((60, 28), (30, 30)), input_shape=input_shape))

    # Normalization
    model.add(Lambda(lambda x: (x / 127.5) - 1.0))

    model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
    model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))

    model.add(Dense(1))
```

The architecture is:
 1. Cropping layer which crops out extraneous parts of the images leaving only the road. *See image below*
 2. Normalization layer which normalizes the images to a valuse between -1 to 1
 3. Three covolutional layers with a 5 x 5 kernel and 2 x 2 strides
 4. Two convolutional layers with a 3 x 3 kernel and  1 x 1 strides
 5. Dropout layer with 20% dropout.
 6. Flattened output from the convolutions
 7. Four fully connected layers.
  
All convolutional and fully connected layers use a ReLU activation function to introduce non-linearity except the 
final output layer.

The image below is the output of the cropping layer.  It's clearly seen that the model will only consider the road 
because the surrounding scenery has been cropped out.

![alt text][image4]

####2. Attempts to reduce overfitting in the model

The model containes a single dropout layer to reduce the risk of over fitting.  Additionally, adding more distinct input samples
and reducing the number of training epochs also lowers the likelihood of over fitting.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.  See the code snippet below 
found on line 214 of `model.py`

```
    model.compile( optimizer='adam', loss='mean_squared_error')
```

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, 
and recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My strategy was start with a model that had been used in the past for analyzing images.  I started with LeNet because 
it was relatively simple.  I modified it to do a regression instead of classification.  After trying it out, I decided
to give the NVidia model a try since it was reported to work well for this purpose.  The implementation was 
fairly straightforward using Keras.  Although not specifically called out in the paper by NVidia, I decided to add a 
dropout layer following the convolutional layers to reduce the likelihood of overfitting.

####2. Final Model Architecture

The final model architecture is implemented in the function `nvidia_model` in `model.py` around line 26

####3. Creation of the Training Set & Training Process

To create a training set, I first drove a one lap around the track and trained the model on this for a few epochs.  It 
soon became evident that more data was needed because the car tended to turn left.  Additionally, the car would tend 
to get itself into situations it could not recover from.  In the hopes of getting the model to generalize a little 
better, I added to the training set by driving the car around the track in the opposite direction.  This helped 
somewhate but I still found that there were situations the car did not recover from.  Next I modified the code to 
augment the training set.  This is described in detail in the next sections.  Finally I added recovery scenarios to the 
training set by recording specific recovery maneuvers at problem areas of the track.  In then end, I recorded what seemed 
to be the equivalent of lap of recovery data.

Below is a sample image of normal center driving

![alt text][image1]

Below is a sample image going backward around the track

![alt text][image2]

Below is a sample image of data collected at one of the problem ares.  

![alt text][image3]

In this case the case of the image above, the car sometimes drove off into the off-road area. 

####4. Data Augmentation
To augment the dataset, I used both left and right camera locations. A correction factor was added or subtracted to or from
the corresponding center steering angle.  The hope here was to encourage the car to steer toward the center if it were 
off center.  In addition I tried to encourage the car to recover from situations where the nose of the car was pointed off the road.
To do this, I tride to simulate the car aiming off the road by by shearing the images to the left or right. 
For these images I also applied steering correction to the center angle angle.  The values for these steering angle corrections were
determined experimentally.  The code for this can be found in the `generator()` function in `model.py` beginning at
line 87.

Bringing it all together, the dataset was randomly shuffled and split 80% / 20% training data to validation data.
Using this I repeatedly trained the model until I determined a suitable number of training epochs. I eventually 
settled on 6 epochs.

