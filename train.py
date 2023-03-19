
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


import wandb
wandb.login()

import argparse
parser = argparse.ArgumentParser(description='Run my NeuralNetwork function')
parser.add_argument('-wp','--wandb_project', default="CS6910_Assignment-1", type=str, help=' ')
parser.add_argument('-we','--wandb_entity', default="CS22M013", type=str, help='')
parser.add_argument('-d','--dataset', default="fashion_mnist", type=str,choices= ["mnist", "fashion_mnist"], help=' ')
parser.add_argument('-e','--epochs', default=10, type=int, help=' ')
parser.add_argument('-b','--batch_size', default=32, type=int, help=' ')
parser.add_argument('-o','--optimizer', default="adam", type=str,choices= ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help=' ')
parser.add_argument('-l','--loss', default="cross_entropy", type=str,choices= ["mean_squared_error", "cross_entropy"], help=' ')
parser.add_argument('-lr','--learning_rate', default=0.001, type=float, help='')
parser.add_argument('-m','--momemtum', default=0.9, type=float, help=' ')
parser.add_argument('-beta','--beta', default=0.9, type=float, help=' ')
parser.add_argument('-beta1','--beta1', default=0.9, type=float, help=' ')
parser.add_argument('-beta2','--beta2', default=0.99, type=float, help=' ')
parser.add_argument('-eps','--epsilon', default=0.00001, type=float, help=' ')
parser.add_argument('-w_d','--weight_decay', default=0.005, type=float, help=' ')
parser.add_argument('-w_i','--weight_init', default="Xavier", type=str,choices= ["random", "Xavier"], help=' ')
parser.add_argument('-nhl','--num_layers', default=3, type=int, help=' ')
parser.add_argument('-sz','--hidden_size', default=64 , type=int, help=' ')
parser.add_argument('-a','--activation', default="ReLU", type=str,choices= ["identity", "sigmoid", "tanh", "ReLU"], help=' ')
args = parser.parse_args()


wandb.init(project=args.wandb_project,entity=args.wandb_entity)
if(args.dataset=='fashion_mnist'):
  data = keras.datasets.fashion_mnist
else:
  data=keras.datasets.mnist


if(args.loss=='mean_squared_error'):
  args.loss='mse'
if(args.weight_init=='Xavier'):
  args.weight_init='xavier'
if(args.activation=='ReLU'):
  args.activation='relu'


#Use the standard train/test split of fashion_mnist (use (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()). 
# Keep 10% of the training data aside as validation data for this hyperparameter search

(X, y), (test_images, test_labels) = data.load_data()
train_images, val_images, train_labels, val_labels = train_test_split(X, y, test_size=0.1, random_state=42)


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
k=len(class_names)

#Flatten the Training, Testing & Validation
image_size=28*28
def flatten_input():
  train_images_X=np.zeros(shape=(len(train_images),image_size),dtype='float32')  #60000x728
  test_images_X=np.zeros(shape=(len(test_images),image_size),dtype='float32')    #10000x728
  val_images_X=np.zeros(shape=(len(val_images),image_size),dtype='float32') 
  for i in range(len(train_images)):
    train_images_X[i]=train_images[i].flatten()/255
  for i in range(len(test_images)):
    test_images_X[i]=test_images[i].flatten()/255
  for i in range(len(val_images)):
    val_images_X[i]=val_images[i].flatten()/255
  return train_images_X,test_images_X,val_images_X

train_images_X,test_images_X,val_images_X=flatten_input()

#Helper Function to one_hot encode the labels.
def one_hot(i):
  y=np.zeros((10,1))
  y[i]=1
  return y

#Activation Functions
def relu(x):
    return np.maximum(0,x)

def tanh(x):
    return np.tanh(x)

def sigmoid(x): 
   return 1./(1.+np.exp(-x))

def derivative_relu(x):
  return 1*(x>0) 

def softmax(x):
  x=x-max(x)
  return np.exp(x)/np.sum(np.exp(x))

def identity(x):
  return x

def derivative_identity(x):
  return np.ones(x.shape)

def derivative_sigmoid(x):
  return sigmoid(x)*(np.ones_like(x)-sigmoid(x))

def derivative_tanh(x):
  return (1 - (np.tanh(x)**2))

def derivative_softmax(x):
  return softmax(x) * (1-softmax(x))


#Loss function for MSE && Cross Entroy with L2 Regularization
def l2loss(parameters,lamda):
  l2loss=0
  for i in range(1,len(parameters)//2+1):
    l2loss+=(lamda/2)*np.sum(np.linalg.norm(parameters['W'+str(i)])**2)
  return l2loss

def loss_function(loss_type,true_output,predicted_output,parameters,lamda):
  if(loss_type=='cross_entropy'):
    return -1.0*np.sum(true_output*np.log(predicted_output+1e-9))
  if(loss_type=='mse'):
    return (1/2) * np.sum((true_output-predicted_output)**2)

def activation_function(funct,x,derivative=False):
  if derivative==True:
    if funct=='softmax':
      return derivative_softmax(x)
    if funct=="sigmoid":
      return derivative_sigmoid(x)
    if funct=="relu":
      return derivative_relu(x)
    if funct=="tanh":
      return derivative_tanh(x)
    if funct=="identity":
      return derivative_identity(x)
  
  else:
    if funct=='softmax':
      return softmax(x)
    if funct=="sigmoid":
      return sigmoid(x)
    if funct=="relu":
      return relu(x)
    if funct=="tanh":
      return tanh(x)
    if funct=="identity":
      return identity(x)

#returns predictions and accuracy for (images,labels).
def find_pred(X,y,params,number_hidden_layers,hidden_layer_size,k,function):
    y_pred=[]
    cnt=0
    for i in range(len(X)):
        predicted_y,activation,pre_activation=feed_forward(number_hidden_layers,params,hidden_layer_size,k,X[i],function)
        y_pred.append(np.argmax(predicted_y))
        if(np.argmax(predicted_y)==y[i]):
          cnt+=1
    accuracy=(cnt/len(X))*100
    return y_pred,accuracy

#returs Validation loss & Validation accuracy for Validation Set.
def validationloss(X,y,params,lamda,number_hidden_layers,hidden_layer_size,function,loss_type):
  cnt,loss=0,0
  for i in range(len(X)):
    predicted_y,activation,pre_activation=feed_forward(number_hidden_layers,params,hidden_layer_size,k,X[i],function)
    loss+=loss_function(loss_type,one_hot(y[i]),predicted_y,params,lamda)
    if(np.argmax(predicted_y)==y[i]):
      cnt+=1
  loss+=l2loss(params,lamda)
  Validation_loss=loss/len(X)  
  Validation_accuracy=cnt/len(X)*100
  return Validation_loss,Validation_accuracy

#helper function for logging in WandB

def calculate(loss,X,y,parameters,number_hidden_layers,hidden_layer_size,k,function,lamda,epoch,loss_type):
  Training_loss=loss/len(X)
  y_pred,Training_accuracy=find_pred(X,y,parameters,number_hidden_layers,hidden_layer_size,k,function)
  Validation_loss,Validation_accuracy=validationloss(val_images_X,val_labels,parameters,lamda,number_hidden_layers,hidden_layer_size,function,loss_type)
  
  print('Epoch:',epoch)
  print('Training loss:',Training_loss)
  print('Training_accuracy',Training_accuracy)
  print('Validation loss:', Validation_loss)
  print('Validation accuracy:',Validation_accuracy)
  wandb.log({'Training_accuracy':Training_accuracy,'Epoch':epoch,'Training_loss':Training_loss,'Validation_loss':Validation_loss,'Validation_accuracy':Validation_accuracy})


# Helper functions for initialization
def init(n_in, n_out,initialization):
    if(initialization=='random'):
      return np.random.default_rng().uniform(low=-0.69,high=0.69,size=(n_in,n_out))
    if(initialization=='xavier'):
        return  np.random.randn(n_in,n_out)*np.sqrt(2/(n_in+n_out))
    if(initialization=='zero'):
      return np.zeros((n_in,n_out))

#initializes parameters w.r.t to number of layers & initialization provided.
def initialize_parameters(number_of_neurons,number_hidden_layers,k,layers,initialization):
  parameters={}
  parameters['W'+str(1)]=init(layers[0],image_size,initialization)
  parameters['b'+str(1)]=init(layers[0],1,initialization)
  for i in range(1,number_hidden_layers):
    parameters['W'+str(i+1)]=init(layers[i],layers[i-1],initialization)
    parameters['b'+str(i+1)]=init(layers[i],1,initialization)
  parameters['W'+str(number_hidden_layers+1)]=init(k,layers[-1],initialization)
  parameters['b'+str(number_hidden_layers+1)]=init(k,1,initialization)
  return parameters

# Implement a feedforward neural network which takes images from the fashion-mnist data as input and outputs a probability distribution over the 10 classes.

def feed_forward(number_hidden_layers,parameters,hidden_layer_size,k,data,function):
  activation={}
  pre_activation={}

  activation['h0']=data.reshape(784,1)
  for i in range(1,number_hidden_layers+1):
    a=np.add(parameters['b'+str(i)],np.matmul(parameters['W'+str(i)],activation['h'+str(i-1)]))
    h=activation_function(function,a)
    #print(h.shape)
    #h=sigmoid(a)
    pre_activation['a'+str(i)]=a
    activation['h'+str(i)]=h
  
  a=np.add(parameters['b'+str(number_hidden_layers+1)],np.matmul(parameters['W'+str(number_hidden_layers+1)],activation['h'+str(number_hidden_layers)]))
  h=softmax(a)
  pre_activation['a'+str(number_hidden_layers+1)]=a
  activation['h'+str(number_hidden_layers+1)]=h
  
  return h,activation,pre_activation

#h:: activation a=:preactivation
#Implement the backpropagation algorithm with support for the following optimisation functions
def back_propogation(parameters,activation,pre_activation,X,y,number_hidden_layers,predicted_y,k,lamda,function,loss_type):

  gradient_parameters,gradient_activation,gradient_preactivation={},{},{}

  #compute output gradient
  e_y=np.zeros((k,1))
  e_y[y][0]=1
  if(loss_type=='cross_entropy'):
    gradient_preactivation['a'+str(number_hidden_layers+1)]=-(e_y-predicted_y)
  else:
    gradient_preactivation['a'+str(number_hidden_layers+1)]=(predicted_y-e_y)*activation_function('softmax',pre_activation['a'+str(number_hidden_layers+1)],True)

  for t in range(number_hidden_layers+1,0,-1):
    #compute gradients w.r.t parameters
    gradient_parameters['W'+str(t)]=np.matmul(gradient_preactivation['a'+str(t)],activation['h'+str(t-1)].T) 
    gradient_parameters['b'+str(t)]=gradient_preactivation['a'+str(t)]
    #print(t,gradient_parameters['W'+str(t)])
    if(t==1):break
    #compute gradients w.r.t layers below
    gradient_activation['h'+str(t-1)]=np.matmul(parameters['W'+str(t)].T,gradient_preactivation['a'+str(t)])

    #compute gradients w.r.t preactivation layer
    gradient_preactivation['a'+str(t-1)]=np.multiply(gradient_activation['h'+str(t-1)],activation_function(function,pre_activation['a'+str(t-1)],True))
  #print(pre_activation)
  return gradient_parameters


#Helper functions for update Rules in optimization function


def update_parameters(parameters,gradient_change,learning_rate):
  for i in range(1,len(parameters)//2+1):
    parameters['W'+str(i)]=parameters['W'+str(i)]-learning_rate*gradient_change['W'+str(i)]
    parameters['b'+str(i)]=parameters['b'+str(i)]-learning_rate*gradient_change['b'+str(i)]
  return parameters

def update_parameters_momentum(parameters,gradient_change,prior_updates,learning_rate,beta):
  for i in range(1,len(parameters)//2+1):
    prior_updates['W'+str(i)]=beta*prior_updates['W'+str(i)]+gradient_change['W'+str(i)]
    parameters['W'+str(i)]=parameters['W'+str(i)]-learning_rate*prior_updates['W'+str(i)]

    prior_updates['b'+str(i)]=beta*prior_updates['b'+str(i)]+gradient_change['b'+str(i)]
    parameters['b'+str(i)]=parameters['b'+str(i)]-learning_rate*prior_updates['b'+str(i)]
  return parameters,prior_updates

def update_parameters_rmsprop(parameters,gradient_change,prior_updates,learning_rate,beta):
  epsilon=1e-9
  for i in range(1,len(parameters)//2+1):
    prior_updates['W'+str(i)]=beta*prior_updates['W'+str(i)]+(1-beta)*(gradient_change['W'+str(i)])**2
    parameters['W'+str(i)]=parameters['W'+str(i)]-gradient_change['W'+str(i)]*(learning_rate/np.sqrt(prior_updates['W'+str(i)]+epsilon))

    prior_updates['b'+str(i)]=beta*prior_updates['b'+str(i)]+(1-beta)*(gradient_change['b'+str(i)])**2
    parameters['b'+str(i)]=parameters['b'+str(i)]-gradient_change['b'+str(i)]*(learning_rate/np.sqrt(prior_updates['b'+str(i)]+epsilon))
  return parameters,prior_updates

#Nesterov Accelerated Gradient Descent
def gradient_descent_nag(number_hidden_layers,hidden_layer_size,batch_size,max_epochs,X,y,k,optimizer,learning_rate,beta,layers,initialization,lamda,function,loss_type):
  
  parameters=initialize_parameters(hidden_layer_size,number_hidden_layers,k,layers,initialization)
  prior_updates=initialize_parameters(hidden_layer_size,number_hidden_layers,k,layers,'zero')
  updates=initialize_parameters(hidden_layer_size,number_hidden_layers,k,layers,'zero')
  temp=initialize_parameters(hidden_layer_size,number_hidden_layers,k,layers,'zero')
  for epoch in range(max_epochs):
    loss,cnt=0,0
    for it in range(1,len(parameters)//2+1):
      updates['W'+str(it)]=prior_updates['W'+str(it)]*beta
      updates['b'+str(it)]=prior_updates['b'+str(it)]*beta
    
    for it in range(1,len(parameters)//2+1):
      temp['W'+str(it)]=parameters['W'+str(it)]-updates['W'+str(it)]
      temp['b'+str(it)]=parameters['b'+str(it)]-updates['b'+str(it)]
    
    for i in range(len(X)):
      predicted_y,activation,pre_activation=feed_forward(number_hidden_layers,temp,hidden_layer_size,k,X[i],function)
      loss+=loss_function(loss_type,one_hot(y[i]),predicted_y,parameters,lamda)
      gradient_parameters=back_propogation(temp,activation,pre_activation,X[i],y[i],number_hidden_layers,predicted_y,k,lamda,function,loss_type)
      
      if(cnt==0):
        gradient_change=initialize_parameters(hidden_layer_size,number_hidden_layers,k,layers,'zero')
      else:
        for iter in range(1,len(parameters)//2+1):
          gradient_change['W'+str(iter)]+=gradient_parameters['W'+str(iter)]
          gradient_change['b'+str(iter)]+=gradient_parameters['b'+str(iter)]
      cnt+=1

      if(cnt%batch_size==0 or i==len(X)-1):
        if(lamda!=0):
          for it in range(1,len(parameters)//2+1):
            gradient_change['W'+str(it)]+=np.dot(lamda,parameters['W'+str(it)])

        t=cnt if (i==len(X)-1) else batch_size
        for it in range(1,len(parameters)//2+1):
          gradient_change['W'+str(it)]=gradient_change['W'+str(it)]/t
          gradient_change['b'+str(it)]=gradient_change['b'+str(it)]/t
        cnt=0

        #update rule
        for it in range(1,len(parameters)//2+1):
          updates['W'+str(it)]=beta*prior_updates['W'+str(it)]+learning_rate*gradient_change['W'+str(it)]
          parameters['W'+str(it)]=parameters['W'+str(it)]-updates['W'+str(it)]
          temp['W'+str(it)]=parameters['W'+str(it)]
          prior_updates['W'+str(it)]=updates['W'+str(it)]

          updates['b'+str(it)]=beta*prior_updates['b'+str(it)]+learning_rate*gradient_change['b'+str(it)]
          parameters['b'+str(it)]=parameters['b'+str(it)]-updates['b'+str(it)]
          temp['b'+str(it)]=parameters['b'+str(it)]
          prior_updates['b'+str(it)]=updates['b'+str(it)]
    l2regularizedloss=l2loss(parameters,lamda)
    
    loss+=l2regularizedloss
    calculate(loss,X,y,parameters,number_hidden_layers,hidden_layer_size,k,function,lamda,epoch,loss_type)

  return parameters


#Stochastic Gradient Descent
def gradient_descent_sgd(number_hidden_layers,hidden_layer_size,batch_size,max_epochs,X,y,k,optimizer,learning_rate,beta,layers,initialization,lamda,function,loss_type):
  parameters=initialize_parameters(hidden_layer_size,number_hidden_layers,k,layers,initialization)
  
  for epoch in range(max_epochs):
    loss=0
    cnt=0
    for i in range(len(X)):
      predicted_y,activation,pre_activation=feed_forward(number_hidden_layers,parameters,hidden_layer_size,k,X[i],function)
      loss+=loss_function(loss_type,one_hot(y[i]),predicted_y,parameters,lamda)
      #print(loss)
      gradient_parameters=back_propogation(parameters,activation,pre_activation,X[i],y[i],number_hidden_layers,predicted_y,k,lamda,function,loss_type)
    
      if(cnt==0):
        gradient_change={}
        gradient_change=gradient_parameters.copy()

      else:
        for iter in range(1,len(parameters)//2+1):
          gradient_change['W'+str(iter)]+=gradient_parameters['W'+str(iter)]
          gradient_change['b'+str(iter)]+=gradient_parameters['b'+str(iter)]
      cnt+=1
      
      if(cnt%batch_size==0 or i==len(X)-1):
        t=cnt if (i==len(X)-1) else batch_size

        if(lamda!=0):
           for it in range(1,len(parameters)//2+1):
             gradient_change['W'+str(it)]+=np.dot(lamda,parameters['W'+str(it)])

        for it in range(1,len(parameters)//2+1):
          gradient_change['W'+str(it)]=gradient_change['W'+str(it)]/t
          gradient_change['b'+str(it)]=gradient_change['b'+str(it)]/t
        cnt=0
        parameters=update_parameters(parameters,gradient_change,learning_rate)
    l2regularizedloss=l2loss(parameters,lamda)
    l2regularizedloss/=len(X)
    calculate(loss,X,y,parameters,number_hidden_layers,hidden_layer_size,k,function,lamda,epoch,loss_type)


  return parameters

#Momentum
def gradient_descent_momentum(number_hidden_layers,hidden_layer_size,batch_size,max_epochs,X,y,k,optimizer,learning_rate,beta,layers,initialization,lamda,function,loss_type):
  parameters=initialize_parameters(hidden_layer_size,number_hidden_layers,k,layers,initialization)
  
  for epoch in range(max_epochs):
    loss=0
    cnt=0
    for i in range(len(X)):
      predicted_y,activation,pre_activation=feed_forward(number_hidden_layers,parameters,hidden_layer_size,k,X[i],function)
      #print(activation['h4'])
      
      loss+=loss_function(loss_type,one_hot(y[i]),predicted_y,parameters,lamda)
      gradient_parameters=back_propogation(parameters,activation,pre_activation,X[i],y[i],number_hidden_layers,predicted_y,k,lamda,function,loss_type)
      if( epoch==0 and i==0):
        #initialize with zero
        prior_updates=initialize_parameters(hidden_layer_size,number_hidden_layers,k,layers,'zero')

      if(cnt==0):
        gradient_change={}
        gradient_change=gradient_parameters.copy()

      else:
        for iter in range(1,len(parameters)//2+1):
          gradient_change['W'+str(iter)]+=gradient_parameters['W'+str(iter)]
          gradient_change['b'+str(iter)]+=gradient_parameters['b'+str(iter)]
      cnt+=1
      
      if(cnt%batch_size==0 or i==len(X)-1):
        t=cnt if (i==len(X)-1) else batch_size

        
        if(lamda!=0):
          for it in range(1,len(parameters)//2+1):
            gradient_change['W'+str(it)]+=np.dot(lamda,parameters['W'+str(it)])

        for it in range(1,len(parameters)//2+1):
          gradient_change['W'+str(it)]=gradient_change['W'+str(it)]/t
          gradient_change['b'+str(it)]=gradient_change['b'+str(it)]/t
        cnt=0
        parameters,prior_updates=update_parameters_momentum(parameters,gradient_change,prior_updates,learning_rate,beta)
    l2regularizedloss=l2loss(parameters,lamda)
    loss+=l2regularizedloss

    calculate(loss,X,y,parameters,number_hidden_layers,hidden_layer_size,k,function,lamda,epoch,loss_type)
  return parameters


#RMSprop
def gradient_descent_rmsprop(number_hidden_layers,hidden_layer_size,batch_size,max_epochs,X,y,k,optimizer,learning_rate,beta,layers,initialization,lamda,function,loss_type):
  parameters=initialize_parameters(hidden_layer_size,number_hidden_layers,k,layers,initialization)
  
  for epoch in range(max_epochs):
    loss=0
    cnt=0
    for i in range(len(X)):
      predicted_y,activation,pre_activation=feed_forward(number_hidden_layers,parameters,hidden_layer_size,k,X[i],function)
      #print(activation['h4'])
      
      loss+=loss_function(loss_type,one_hot(y[i]),predicted_y,parameters,lamda)
      gradient_parameters=back_propogation(parameters,activation,pre_activation,X[i],y[i],number_hidden_layers,predicted_y,k,lamda,function,loss_type)
      if( epoch==0 and i==0):
        #initialize with zero
        prior_updates=initialize_parameters(hidden_layer_size,number_hidden_layers,k,layers,'zero')

      if(cnt==0):
        gradient_change={}
        gradient_change=gradient_parameters.copy()

      else:
        for iter in range(1,len(parameters)//2+1):
          gradient_change['W'+str(iter)]+=gradient_parameters['W'+str(iter)]
          gradient_change['b'+str(iter)]+=gradient_parameters['b'+str(iter)]
      cnt+=1
      
      if(cnt%batch_size==0 or i==len(X)-1):
        t=cnt if (i==len(X)-1) else batch_size

        
        if(lamda!=0):
          for it in range(1,len(parameters)//2+1):
            gradient_change['W'+str(it)]+=np.dot(lamda,parameters['W'+str(it)])

        for it in range(1,len(parameters)//2+1):
          gradient_change['W'+str(it)]=gradient_change['W'+str(it)]/t
          gradient_change['b'+str(it)]=gradient_change['b'+str(it)]/t
        cnt=0
        parameters,prior_updates=update_parameters_rmsprop(parameters,gradient_change,prior_updates,learning_rate,beta)
    l2regularizedloss=l2loss(parameters,lamda)
    loss+=l2regularizedloss
    
    calculate(loss,X,y,parameters,number_hidden_layers,hidden_layer_size,k,function,lamda,epoch,loss_type)
  return parameters

# Adam & Nadam
def gradient_descent_adam(number_hidden_layers,hidden_layer_size,batch_size,max_epochs,X,y,k,optimizer,learning_rate,beta,layers,initialization,lamda,function,loss_type,beta1,beta2):
  
  parameters=initialize_parameters(hidden_layer_size,number_hidden_layers,k,layers,initialization)
  momentum=initialize_parameters(hidden_layer_size,number_hidden_layers,k,layers,'zero')
  momentum_hat=initialize_parameters(hidden_layer_size,number_hidden_layers,k,layers,'zero')
  update=initialize_parameters(hidden_layer_size,number_hidden_layers,k,layers,'zero')
  update_hat=initialize_parameters(hidden_layer_size,number_hidden_layers,k,layers,'zero')
  for epoch in range(max_epochs):
    loss=0
    cnt=0
    epsilon=1e-10
    for i in range(len(X)):
      predicted_y,activation,pre_activation=feed_forward(number_hidden_layers,parameters,hidden_layer_size,k,X[i],function)
      loss+=loss_function(loss_type,one_hot(y[i]),predicted_y,parameters,lamda)
      gradient_parameters=back_propogation(parameters,activation,pre_activation,X[i],y[i],number_hidden_layers,predicted_y,k,lamda,function,loss_type)
      if(cnt==0):
        gradient_change={}
        gradient_change=gradient_parameters.copy()

      else:
        for iter in range(1,len(parameters)//2+1):
          gradient_change['W'+str(iter)]+=gradient_parameters['W'+str(iter)]
          gradient_change['b'+str(iter)]+=gradient_parameters['b'+str(iter)]
      cnt+=1
      
      if(cnt%batch_size==0 or i==len(X)-1):

        if(lamda!=0):
          for it in range(1,len(parameters)//2+1):
            gradient_change['W'+str(it)]+=np.dot(lamda,parameters['W'+str(it)])

        t=cnt if (i==len(X)-1) else batch_size
        for it in range(1,len(parameters)//2+1):
          gradient_change['W'+str(it)]=gradient_change['W'+str(it)]/t
          gradient_change['b'+str(it)]=gradient_change['b'+str(it)]/t
        
        cnt=0
        for it in range(1,len(parameters)//2+1):
          momentum['W'+str(it)]=beta1*momentum['W'+str(it)]+(1-beta1)*gradient_change['W'+str(it)]
          momentum['b'+str(it)]=beta1*momentum['b'+str(it)]+(1-beta1)*gradient_change['b'+str(it)]
          momentum_hat['W'+str(it)]=momentum['W'+str(it)]/(1-beta1**(epoch+1))
          momentum_hat['b'+str(it)]=momentum['b'+str(it)]/(1-beta1**(epoch+1))

        for it in range(1,len(parameters)//2+1):
          update['W'+str(it)]=beta2*update['W'+str(it)]+(1-beta2)*gradient_change['W'+str(it)]**2
          update['b'+str(it)]=beta2*update['b'+str(it)]+(1-beta2)*gradient_change['b'+str(it)]**2
          update_hat['W'+str(it)]=update['W'+str(it)]/(1-beta2**(epoch+1))
          update_hat['b'+str(it)]=update['b'+str(it)]/(1-beta2**(epoch+1))

        if(optimizer=='adam'):
          for it in range(1,len(parameters)//2+1):
            parameters['W'+str(it)]=parameters['W'+str(it)]-(learning_rate*momentum_hat['W'+str(it)]/np.sqrt(update_hat['W'+str(it)]+epsilon))
            parameters['b'+str(it)]=parameters['b'+str(it)]-(learning_rate*momentum_hat['b'+str(it)]/np.sqrt(update_hat['b'+str(it)]+epsilon))
        else:
          #update rule for nadam  
          for it in range(1,len(parameters)//2+1):
            parameters['W'+str(it)]=parameters['W'+str(it)]-(learning_rate/np.sqrt(update_hat['W'+str(it)]+epsilon))*(beta1*momentum_hat['W'+str(it)]+(1-beta1)*gradient_change['W'+str(it)]/(1-beta1**(epoch+1)))
            parameters['b'+str(it)]=parameters['b'+str(it)]-(learning_rate/np.sqrt(update_hat['b'+str(it)]+epsilon))*(beta1*momentum_hat['b'+str(it)]+(1-beta1)*gradient_change['b'+str(it)]/(1-beta1**(epoch+1)))
    
    l2regularizedloss=l2loss(parameters,lamda)
    loss+=l2regularizedloss

    calculate(loss,X,y,parameters,number_hidden_layers,hidden_layer_size,k,function,lamda,epoch,loss_type)

  return parameters


#Neural Network function
def NeuralNetwork():
  k=10 
  beta=float(args.beta)
  beta1=float(args.beta1)
  beta2=float(args.beta2)
  epsilon=float(args.epsilon)
  loss_type=args.loss
  number_hidden_layers=args.num_layers
  hidden_layer_size=args.hidden_size
  batch_size=args.batch_size
  max_epochs=args.epochs
  optimizer=args.optimizer
  function=args.activation
  learning_rate=args.learning_rate
  lamda=args.weight_decay
  initialization=args.weight_init
  layers=[hidden_layer_size for i in range(number_hidden_layers)]
  run_name = "lr_{}_ac_{}_in_{}_op_{}_bs_{}_L2_{}_ep_{}_nn_{}_nh_{}_loss_{}".format(learning_rate, function,initialization, optimizer, batch_size, lamda, max_epochs, hidden_layer_size, number_hidden_layers,loss_type)
  print(run_name)

  if(optimizer=='sgd'):
    params=gradient_descent_sgd(number_hidden_layers,hidden_layer_size,batch_size,max_epochs,train_images_X,train_labels,k,optimizer,learning_rate,beta,layers,initialization,lamda,function,loss_type)
  if(optimizer=='nag'):
    params=gradient_descent_nag(number_hidden_layers,hidden_layer_size,batch_size,max_epochs,train_images_X,train_labels,k,optimizer,learning_rate,beta,layers,initialization,lamda,function,loss_type)
  if(optimizer=='momentum'):
    params=gradient_descent_momentum(number_hidden_layers,hidden_layer_size,batch_size,max_epochs,train_images_X,train_labels,k,optimizer,learning_rate,beta,layers,initialization,lamda,function,loss_type)
  if(optimizer=='rmsprop'):
    params=gradient_descent_rmsprop(number_hidden_layers,hidden_layer_size,batch_size,max_epochs,train_images_X,train_labels,k,optimizer,learning_rate,beta,layers,initialization,lamda,function,loss_type)
  if(optimizer=='adam'):
    params=gradient_descent_adam(number_hidden_layers,hidden_layer_size,batch_size,max_epochs,train_images_X,train_labels,k,optimizer,learning_rate,beta,layers,initialization,lamda,function,loss_type,beta1,beta2)
  if(optimizer=='nadam'):
    params=gradient_descent_adam(number_hidden_layers,hidden_layer_size,batch_size,max_epochs,train_images_X,train_labels,k,optimizer,learning_rate,beta,layers,initialization,lamda,function,loss_type,beta1,beta2)
  
  pred_labels,accuracy=find_pred(test_images_X,test_labels,params,number_hidden_layers,hidden_layer_size,k,function)
  print()
  print("Testing Accuracy:",accuracy)
  wandb.run.name = run_name
  wandb.run.save()


NeuralNetwork()

