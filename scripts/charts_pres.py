#!/usr/bin/env python
# coding: utf8

import matplotlib.pyplot as plt

def function1(x):
    return -(x**2)*1.0/10 + 2*x

def function2(x):
    return 11.0*(x**3)/750 - 3.0*x**2/10 + 38.0*x/15

def function3(x):
    return -(x**3)/125+6*x**2/25-13*x/5+20

def function4(x):
    return -(x**3)/375 + x**2/25 + 13*x/15

def charts_explication_goal():
    X1 = []
    X2 = []
    Y1 = []
    Y2 = []

    for i in range(5,38):
        i = i*1.0 /2
        if(i < 9):
            X1.append(i)
            Y1.append(function1(i))
        else:
            X2.append(i)
            Y2.append(function1(i))

    plt.plot(X1,Y1,'-',label='past/current cons')
    plt.plot(X2,Y2,'--',label='prediction cons')
    plt.xlabel('time')
    plt.ylabel('consumption')
    plt.grid(True)
    plt.legend()
    plt.ylim(0,12)
    plt.show()



def generate_plot_functions(begin,end,step,palier):
    X = []
    Y1 = []
    Y2 = []
    Y3 = []
    Y4 = []
    nb_steps = int((end - begin) / step)
    for i in range(nb_steps):
        x = i*step + begin
        X.append(x)
        if(palier != -1):
            Y1.append(function1(x) - function1(x)%palier)
            Y2.append(function2(x) -  function2(x)%palier)
            Y3.append(function3(x) -  function3(x)%palier)
            Y4.append(function4(x) -  function4(x)%palier)
        else:
            Y1.append(function1(x))
            Y2.append(function2(x))
            Y3.append(function3(x))
            Y4.append(function4(x))

    plt.plot(X,Y1,'r-',label='cons1')
    plt.plot(X,Y2,'b-',label='cons2')
    plt.plot(X,Y3,'g-',label='cons3')
    plt.plot(X,Y4,'m-',label='cons4')
    plt.xlabel("time")
    plt.ylabel("consumption")
    plt.legend()
    plt.grid()
    plt.show()


def generate_array(begin,end,step,palier=-1):
    M = []
    nb_steps = int((end - begin) / step)
    for i in range(nb_steps):
        x = i*step + begin
        if(palier != -1):
            M.append([x,int(function1(x) - function1(x)%palier),0])
            M.append([x,int(function2(x) - function2(x)%palier),1])
            M.append([x,int(function3(x) - function3(x)%palier),2])
            M.append([x,int(function4(x) - function4(x)%palier),3])

        else:
            M.append([x,function1(x),0])
            M.append([x,function2(x),1])
            M.append([x,function3(x),2])
            M.append([x,function4(x),3])

    print(M)

    X = []
    Y = []
    s = "new double[,] {"
    for i in range(len(M)):
        s += "{"+str(M[i][0])+","+str(M[i][1])+"},"
        X.append(M[i][0])
        Y.append(M[i][1])
    s = s[:len(s)-1] + "}"

    for i in range(len(M)):
        print(str(M[i][0]) + "&" + str(M[i][1]) + "&" + str(M[i][2]) + "\\\ ")
    #print(s)

    plt.plot(X,Y,'r-',label='global points')
    plt.legend()
    plt.show()



#generate_array(0,20,0.5,4)
#generate_plot_functions(0,20,0.5,0.01)
charts_explication_goal()
