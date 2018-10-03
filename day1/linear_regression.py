import random
import math
def gen(w,b):
    x_true=[]
    y_true=[]
    for i in range(100):
        temp = random.uniform(-3.1,10.2)
        x_true.append(temp)
        y_true.append(w*temp+b)
    return x_true,y_true

def linear(x_true,y_true,lr=0.01,epoch=5):
    w = 0
    b = 0

    for j in range(epoch):
        grad_b = 0
        grad_w = 0
        loss = 0
        for i in range(len(x_true)):
            y = w*x_true[i]+b
            loss += (y-y_true[i])*(y-y_true[i])
            grad_b = grad_b + (y-y_true[i])
            grad_w = grad_w + (y-y_true[i])*x_true[i]
            
            # print(loss,grad_b,grad_w)
        grad_b /=len(x_true)
        grad_w /=len(x_true)
        w = w - lr*grad_w
        b = b - lr*grad_b  

        if j < 1000:
                # print(loss)
                print(loss)
    print(w,b)
x_true,y_true = gen(10,3)
linear(x_true,y_true,lr=0.0001,epoch=100000)

      

