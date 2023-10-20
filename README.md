# GPT2Domain
This application is interactive system tailored for conversational AI based on a finely-tuned GPT-2 model on a fictional company. With the power of GPT-2, it delivers contextually relevant and coherent responses to users' queries. But can generate inappropriate responses that are unrelated.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

# Features
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

## Dataset - open the train folder:
Remember to create and add your training and validation data to the train folder - i have included some sample data i based on a fictional company that sells timber - I used a trainign and validation set ratio of 80/20 in this sample, but you can adjust the content as needed.
1. Add your training dataset labeled as train.txt (or use sample)
2. Add your validation dataset labeled as valid.txt (or use sample)

## Train your model
Once you have added your dataset to the train folder, remember to train your model (this can take sometime so ensure you have enough memory and CPU power - you can adjust the epochs, train and eval batch sizes to combat low memory but it wil take days to complete)
1. Create a folder called model if it does not exist (included with sample)
2. Navigate to where you cloned the repository on your local machine that contains the train_domain.py file and run python3 train_domain.py
3. Result on sample data
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 738/738 [00:00<00:00, 3417.10 examples/s]
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 346/346 [00:00<00:00, 3698.34 examples/s]
{'loss': 5.1901, 'learning_rate': 5.000000000000001e-07, 'epoch': 0.08}                                                                                       
{'loss': 4.2764, 'learning_rate': 8.636363636363637e-06, 'epoch': 4.17}                                                                                       
{'loss': 3.1341, 'learning_rate': 6.363636363636364e-06, 'epoch': 8.33}                                                                                       
{'eval_loss': 3.382888078689575, 'eval_runtime': 27.6239, 'eval_samples_per_second': 12.525, 'eval_steps_per_second': 0.217, 'epoch': 8.33}                   
{'loss': 2.5759, 'learning_rate': 4.0909090909090915e-06, 'epoch': 12.5}                                                                                                
{'loss': 2.2649, 'learning_rate': 1.8181818181818183e-06, 'epoch': 16.67}                                                                                               
{'eval_loss': 3.026434898376465, 'eval_runtime': 47.8554, 'eval_samples_per_second': 7.23, 'eval_steps_per_second': 0.125, 'epoch': 16.67}                              
 95%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋      | 228/240 [2:18:15<14:27, 72.28s/it]
 98%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍   | 235/240 [2:27:39<07:49, 93.87s/it]{'train_runtime': 9194.3065, 'train_samples_per_second': 1.605, 'train_steps_per_second': 0.026, 'train_loss': 2.9188002705574037, 'epoch': 20.0}                                                                  
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240

## Confirm the modle was saved
Once your model has been trained, verify that the model and tokenizer has been saved to the folder called model.

## Run the application
From the terminal run phthon3 elastic_gpt2.py, if everything worked as planned the flask server should start and you should be able to access the application chat window

## Chat
1. Open project folder and open chat.html, the current response settings are very deterministic (its very focused) it also leans towards the training data a little too much!!

# Contributing
Contributions are welcome!

# License
This project is licensed under the MIT License. See the `LICENSE` file for details.
