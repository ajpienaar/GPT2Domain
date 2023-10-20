# GPT2Domain
This application is interactive system tailored for conversational AI based on a finely-tuned GPT-2 model on a fictional company. With the power of GPT-2, it delivers contextually relevant and coherent responses to users' queries. But can generate inappropriate responses that are unrelated.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Features
1. **Fine-Tuned GPT-2**: The core of this application is a GPT-2 model, which has been fine-tuned to provide precise and relevant answers.
2. **Web Integration**: It facilitates seamless interactions via a dedicated API endpoint using flask. Users can send questions as JSON payloads to receive corresponding answers.
3. **Interactive UI**: The application features a simplistic chat interface (chat.html), providing an intuitive user experience in a typewriter style response.


# Installation Instructions for Setting Up the Application
1. Clone the repository to your project directory.
2. Navigate to the cloned directory

## Update and Upgrade Your System
Before starting the installation, it's good practice to update and upgrade your system to ensure you have the latest packages.
1. sudo apt update
2. sudo apt upgrade

## Install Python 3 and pip
Lets install pythion and pip
1. sudo apt install python3-pip

## Install Required Python Libraries
With pip installed, proceed to install the necessary Python libraries for the application.
1. pip install transformers
2. pip install datasets
3. pip install torch torchvision
4. pip install tensorboard

## Update and Install Accelerate
Ensure you have the latest version of the accelerate library.
1. pip install accelerate -U

## Install Flask server
1. pip install Flask


# Configuration
Clone this repository to your project folder

### Dataset - open the train folder:
Remember to create and add your training and validation data to the train folder - i have included some sample data i based on a fictional company that sells timber - I used a trainign and validation set ratio of 80/20 in this sample, but you can adjust the content as needed.
1. Add your training dataset labeled as train.txt (or use sample)
2. Add your validation dataset labeled as valid.txt (or use sample)

### Train your model
Once you have added your dataset to the train folder, remember to train your model (this can take sometime so ensure you have enough memory and CPU power - you can adjust the epochs, train and eval batch sizes to combat low memory but it wil take days to complete)
1. Create a folder called model if it does not exist (included with sample)
2. Navigate to where you cloned the repository on your local machine that contains the train_domain.py file and run python3 train_domain.py

## Confirm the modle was saved
Once your model has been trained (if using a new model), verify that the model and tokenizer has been saved to the folder called model.

### Chat
1. Open project folder and open chat.html, the current response settings are vert deterministic (its very focused) it also leans towards the training data a little too much

# Contributing
Contributions are welcome!

# License
This project is licensed under the MIT License. See the `LICENSE` file for details.
