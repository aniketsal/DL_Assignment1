# DL_Assignment1

## Aniket Ravindra Salunke CS22M013

### Packages to import:
- numpy
- wandb
- sklearn
- tensorflow

### This Git contains 2 files

  1. `train.py` which is Python file which we need to run using the following commands as shown in report.
  - For train.py you need to login into your wandb account from the terminal using the api key.
  - In the arguments into the terminal please give valid credentials like project_name and entity_name.
  - Run python file with the following commands and using arguments that you need
  
    `python train.py -e 10 -w_i Xavier -b 32 -o sgd -nhl 3 -sz 64 -d fashion_mnist -l cross_entropy -lr 0.001`
   
    
  2. `DL_Assignment1.ipynb` which is Python notebook with all the questions of the assignment.
  
  - A NeuralNetwork() function needs to be invoked with specific hyperparameters

  - To add the optimizer , we can modify the NeuralNetwork() function.
