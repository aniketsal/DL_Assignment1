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
  - Run python file with the following commands and using arguments from the table.
  
    `python train.py -e 10 -w_i Xavier -b 32 -o sgd -nhl 3 -sz 64 -d fashion_mnist -l cross_entropy -lr 0.001`
     
   |           Name           | Default Value | Description                                                               |
| :----------------------: | :-----------: | :------------------------------------------------------------------------ |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard      |
|  `-we`, `--wandb_entity` |     myname    | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
|     `-d`, `--dataset`    | fashion_mnist | choices:  ["mnist", "fashion_mnist"]                                      |
|     `-e`, `--epochs`     |       1       | Number of epochs to train neural network.                                 |
|   `-b`, `--batch_size`   |       4       | Batch size used to train neural network.                                  |
|      `-l`, `--loss`      | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"]                         |
|    `-o`, `--optimizer`   |      sgd      | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]          |
| `-lr`, `--learning_rate` |      0.1      | Learning rate used to optimize model parameters                           |
|    `-m`, `--momentum`    |      0.5      | Momentum used by momentum and nag optimizers.                             |
|     `-beta`, `--beta`    |      0.5      | Beta used by rmsprop optimizer                                            |
|    `-beta1`, `--beta1`   |      0.5      | Beta1 used by adam and nadam optimizers.                                  |
|    `-beta2`, `--beta2`   |      0.5      | Beta2 used by adam and nadam optimizers.                                  |
|    `-eps`, `--epsilon`   |    0.000001   | Epsilon used by optimizers.                                               |
| `-w_d`, `--weight_decay` |       .0      | Weight decay used by optimizers.                                          |
|  `-w_i`, `--weight_init` |     random    | choices:  ["random", "Xavier"]                                            |
|  `-nhl`, `--num_layers`  |       1       | Number of hidden layers used in feedforward neural network.               |
|  `-sz`, `--hidden_size`  |       4       | Number of hidden neurons in a feedforward layer.                          |
|   `-a`, `--activation`   |    sigmoid    | choices:  ["identity", "sigmoid", "tanh", "ReLU"]                         |

   
    
  2. `DL_Assignment1.ipynb` which is Python notebook with all the questions of the assignment.
  
  - A NeuralNetwork() function needs to be invoked with specific hyperparameters

  - To add the optimizer , we can modify the NeuralNetwork() function.

Report Using WandB:

https://wandb.ai/cs22m013/dl_assignment1/reports/CS22M013-s-CS6910-Assignment-1--VmlldzozODE0MTMw?accessToken=kbg89fnu4id4562em9s87j4ic0v6y2ccp61w5oizpcenm9p84xsqsefer03zvfwe
