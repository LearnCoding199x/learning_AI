import random
import math
import matplotlib.pyplot as plt
def gen(n):
    b = random.random()
    w_true = []
    w_true.append(b)
    for i in range(n):
        w_true.append(random.random())
    x_true = []
    y_true = []
    for i in range(100):
        temp = []
        temp.append(1)
        sum = b
        for j in range(n):
            value = random.random()
            temp.append(value)
            sum+=value*w_true[j+1]
        y_true.append(sum)
        x_true.append(temp)
    print(w_true)
    return x_true,y_true
    
def linear_nvar(x_true,y_true,lr=0.01,iterations=1000):
    w = []
    total_loss = []
    for i in range(len(x_true[0])):
        w.append(0)
    for iter in range(iterations):
        loss = 0
        grad_w = []
        for i in range(len(x_true[0])): #create grad_x(i) 
            grad_w.append(0)
        for i in range(100): #number of examples
            y = 0
            for j in range(len(x_true[i])): #calculate y
                y+=x_true[i][j]*w[j]
            loss+=0.5*(y-y_true[i])*(y-y_true[i])
            for j in range(len(x_true[i])):
                grad_w[j]+=(y-y_true[i])*x_true[i][j]
        loss/=100
        for i in range(len(x_true[0])):
            grad_w[i]/=100
            w[i] -= lr*grad_w[i]
        total_loss.append(loss)
        # print(loss)
    print(w)
    plt.plot(total_loss)
    plt.ylabel('loss')
    plt.show()
x_true,y_true = gen(2)
linear_nvar(x_true,y_true,lr=0.001,iterations=100000)



