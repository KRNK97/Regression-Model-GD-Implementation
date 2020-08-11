# <------------------------- IMPORT LIBRARIES -------------------------->

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# <------------------------- DATA INITIALIZATION ----------------------->

# the number of samples we want to have ( rows )
observations = 150

# we will take 2 input features ( random numpy array ) and generate 1 target
x1 = np.random.uniform(-20,20,size=(observations,1))
x2 = np.random.uniform(-20,20,size=(observations,1))

# combine the inputs into a single matrix for use in dot products
inputs = np.column_stack((x1,x2))

# get the actual targets
targets = -3*x1 + 2*x2 + 120

# we will test a range of learning rates to graph their respective outputs
lr= [0.001,0.012,0.0151]


# <----------------------------- LEARNING --------------------------------->

for rate in lr:
    count = 0
    
    # epochs are the number of passes or iterations we want to perform on the complete dataset
    # the weights and biases are updated on each epoch and new improved weights and biases are generated using the update rule
    EPOCHS = 40                                         
    
    # for each learning rate , we want to have similar initial weights and bias matrices
    weights = np.random.uniform(-1,1,size=(2,1))
    biases = np.random.uniform(-1,1,size=1)
    
    # we will use this list to graph the losses against the number of iterations
    losses = []
    
    # the actual learning begins here -->
    for i in range(EPOCHS):
        
        # get our predicted outputs and measure the delta or difference between actual and predicted values
        outputs = np.dot(inputs,weights) + biases
        deltas = outputs - targets
        
        # we use the l2-norm loss ( sum of squares ) as loss function , we / by observations to get average loss per observation
        loss = np.sum(deltas**2) / 2 / observations
        losses.append(loss)
        
        #print(i,loss)
        count +=1
        #if loss <= 0.1:
            #break
        
        # we get the averge delta value for each observation
        deltas_scaled = deltas / observations

        # update the weights using update rule which is differnt for both biases and weights
        weights = weights - rate*np.dot(inputs.T,deltas_scaled)
        biases = biases - rate*np.sum(deltas_scaled)
        
        
# <------------------------------ VISUALIZATION --------------------------->    

    # at the end of a result , we visualize the output to show the effect of different learning rates 
    plt.plot([x for x in range(count)],losses)
    plt.scatter([x for x in range(count)],losses,lw=0.00001,c='purple')
    plt.ylim(2000,9000)
    plt.xlim(0,20)
    plt.title(rate)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()
    
    
# <-------------------------------- OBSERVATIONS --------------------------->

    # Too slow Learining Rate ----> Wont Converge or will take too long.
    # Good Learning Rate  --------> Will Converge in ideal time.
    # Too fast Learning Rate -----> Wont Converge as it will miss the Minimum point and keep increasing after that.
