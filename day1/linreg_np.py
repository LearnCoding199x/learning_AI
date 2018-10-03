import numpy as np 
import matplotlib.pyplot as plt
def gen(n):
    w_true = np.random.rand(n)
    x_true = np.random.rand(100,n)
    for i in range(100):
        x_true[i,0]=1
    y_true = np.matmul(x_true,w_true)
    print(w_true)
    return x_true,y_true
    
def linear_nvar(x_true,y_true,lr=0.01,iterations=1000):
    n_feature = x_true.shape[1]
    n_data = x_true.shape[0]

    losses = []
    w = np.zeros(n_feature)

    for iter in range(iterations):
        loss = 0

        grad_w = np.zeros(n_feature)

        diff = np.multiply(x_true,w)
        diff = np.sum(diff,axis=1)
        diff = np.subtract(diff,y_true)

        temp = np.matmul(diff,x_true)
        temp = temp/n_data

        grad_w = np.add(grad_w,temp)
        loss = np.sum(0.5*np.power(diff,2))/100
        # for i in range(100): #number of examples
        #     y = 0
        #     for j in range(len(x_true[i])): #calculate y
        #         y+=x_true[i][j]*w[j]
        #     loss+=0.5*(y-y_true[i])*(y-y_true[i])
        #     for j in range(len(x_true[i])):
        #         grad_w[j]+=(y-y_true[i])*x_true[i][j]
        w = np.subtract(w,lr*grad_w)
        losses.append(loss)
        # print(loss)
    print(w)
    plt.plot(losses)
    plt.ylabel('loss')
    plt.show()
x_true,y_true = gen(4)
linear_nvar(x_true,y_true,lr=0.01,iterations=100000)
