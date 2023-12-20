
# Handwritten Character Recognition
Capstone project for the MLZcoomcamp online course. This project focuses on creating a Convolutional Neural Network (CNN) for recognizing handwritten characters. It merges two datasets: one containing handwritten images of alphabets (A-Z) taken from kaggle (see references) ,  and another consisting of handwritten digits (0-9).

The model that is developed here is used to predict images with solid font color, similar to the image example with the name image.png . the code for the python scripts are for png images but if the reader wishes to test another file extension than is necessary to change the file extension in the predict script to the desired one

## Datasets Used
Handwritten Alphabet Dataset: This dataset contains images of handwritten alphabets in grayscale, each centered within a 20x20 pixel box and stored as 28x28 pixel images. The images are sourced from various sources, including NIST and NMIST large datasets. The dataset serves as a resource for beginners in machine learning to develop predictive models for recognizing handwritten characters. Dataset Link

MNIST Dataset: This well-known dataset consists of handwritten digits from 0-9. It's a widely used dataset for developing machine-learning models for image classification tasks.

## Project Overview
The GitHub repository consists on the following files:

1) Notebook: The repository includes a Jupyter Notebook containing Exploratory Data Analysis (EDA), the development of two models, and hyperparameter tuning. The best model achieved an accuracy of 98.570% using 6 epochs and an optimizer with a learning rate of 0.001.

2) Python Files:
* train.py: Used for training the initial model (initial_model.h5), a Final Model(cnn_model.h5) trained on the complete dataset and also creates a TensorFlow Lite model for serving as a web application using Flask.
* predict.py: Uses the TensorFlow Lite model for predicting an image provided by the user.
* predict_char_Test.py: Used to test the pre-uploaded model provided in the repository.

3) requirements.txt: packages list needed for the project

4) initial_model.h5 : Keras model type.the best model achieved for the project but trained on a subset
5) cnn_model.h5 : Keras model type. the best model achieved for the project trained on the complete dataset

6) cnn_model.tflite : tensorflow lite mode, used for serving the model using flask or another webservice

7) image.png: image used for testing the cnn model
   
## Challenges Faced
The project encountered challenges related to reproducibility due to the phenomenon of gradient explosion during CNN training with the dataset. To address this, the optimizer was tuned to mitigate this issue. For user reference, both a Keras model file (initial_model.h5) and a TensorFlow Lite model are provided in the repository for users to evaluate the model's performance.

### Conclusion

In conclusion the CNN model has the following configurations

- Parameters for the model
- learning_rate = 0.001
- epochs = 6
- optimizer = Adam(learning_rate=learning_rate, clipnorm=0.9)
- model = Sequential()
- model.add(Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
- model.add(BatchNormalization()) #Used to avoid 'gradient explosion' as seen in previous experiments with this dataset
- model.add(MaxPooling2D((2, 2)))
- model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
- model.add(Dropout(0.2)) #Used to avoid 'gradient explosion' as seen in previous experiments with this dataset
- model.add(Flatten())
- model.add(Dense(36, activation='softmax')) #36 is the total number of labels in the dataset
- model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

## Train the final model and Predict Using Flask for local deployment

### Prerequisites

Before running the scripts, ensure you have the following dependencies installed:

- Python (version 3.11.6)
- Create virtual environment using:
`python -m venv env`
- Install inside the virtual environment the packages provided in the requirement.txt file

### Running the Code

1. Clone this repository to your local machine.

   ```bash
   git clone https://github.com/your-username/your-repo.git
   ```
2.Create Virtual Enironment

Take the following steps:

- Open the Command Line Interpreter in windows or Bash in Linux & MacOs

- Go to the directory where you want to keep your project using the **cd** command, for example, it can look like this:

```
cd "D:\Documents\zoomcamp\final_project"
```

- Use the following command to call the venv module.

```
python -m venv env
```
- At this point, some directories are created for you. The directory names differ slightly depending on your operating system.

The directories look like this in Windows:

```
/env
  /include
  /lib
  /Scripts
```

The directories look like this in Linux and macOS:

```
/env
  /bin
  /include
  /lib
```

>Your environment needs the env directory to keep track of details like which version of Python and which libraries you're using. Don't put your program files in the env directory. We suggest that you put your files in a directory called src or something similar. The project structure might then look like this:

```
/env
/src
  train.py
```

3. Download the project files and save them in the same folder directory wich you created the virtual environment (e.g. ../src)
- cnn_model.h5
- image.png
- Notebook.ipynb
- predict.py
- prediction_character_test
- requirements.txt
- train.py

4.  Activate the Virtual Environment:

At this point, some directories are created for you. The directory names differ slightly depending on your operating system.

The directories look like this in Windows:

```Output

/env
  /include
  /lib
  /Scripts
```
The directories look like this in Linux and macOS:

```Output

/env
  /bin
  /include
  /lib
```

For Windows:
```
.\venv\Scripts\activate
```

For Linux and MacOS:
```
source venv/bin/activate 
```

3. Install Dependencies
Install the dependencies:
```
pip install -r requirements.txt
```

4. Copy the cnn_model, cnn_model, train.py, predict.py, prediction_character_test.py inside the same directory folder as the virtual environment is created:

For windows:
`Script`

For Linux and MacOs
`bin`

5. Run train.py using the following command assuming we are inside the project directory:
```
python train.py
```

6. For local deployment, start up the Flask server for prediction API:
```
python predict.py
```

Or use a WSGI server, Waitress to run:

`waitress-serve --listen=0.0.0.0:9696 predict:app`

It will run the server on localhost using port 9696.

7. Test the prediction model

- Open a new command Line windows
 
- Activate the virtual environment in this new CLI window

- Execute the prediction_character_test.py file
```
python  prediction_character_test.py
```

8. Send a request to the prediction API http://localhost:9696/predict and get the response:

## Acknowledgments

- Download the original dataset and more information on:
https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format
- Step by step for create virtual environments and manage python projects
https://learn.microsoft.com/en-us/training/modules/python-create-manage-projects/1-introduction

## Contact

Raúl Hernández
- www.raulhernandez.tech
