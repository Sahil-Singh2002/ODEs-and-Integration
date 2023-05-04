#################################################################
## Functions to carry out numerical solution of first order IVPS
#################################################################

#################################################################
## Imports
## - No further imports should be necessary
#################################################################
import numpy as np

#################################################################
## Functions to be completed by student
#################################################################

#%% Q3 code

def adams_bashforth(f,a,b,ya,n,method):
    '''
    The function approximates the solution y(t) for an initial
    value problem, over the interval [a,b], with RHS function given in f and initial
    condition given in ya. Finds this based on method 1 and 2. For method 1 using Euler
    while for method 2 it uses Adams-Bashforth method.

    Parameters
    ----------
    a (float): The lower limit of integration
    b (float): The upper limit of integration
    n (int): The number of subintervals to use
    f (function): The function to find Numerical first order ODE for
    ya (float):The intual condition/ value for y
    method (int) : of only two types 1 representing Euler's Method and 
    2 representing Adam-Bashforth method.

    Returns
    -------
    t (ndarray): The position in the interval [a,b].
    y (ndarray): the point y(t) approximated based on method 1 or 2.

    '''
    #assert
    if type(n) != int or n<= 0:
        raise ValueError("The input needs to be posive Integer")
    #initual set
    y,t = np.zeros(n+1),np.linspace(a,b,n+1)
    diff_n = (b-a)/n
    y[0]=ya
    #Euler = 1
    if method ==1:
        for i in range(n):
            y[i+1] = f(t[i],y[i])*diff_n +y[i]
    #Adam-Bashforth =2
    elif method ==2:
        y[1] = y[0]+ f(t[0],y[0])*diff_n
        for j in range(1,n):
            y[j+1]=  y[j]+  (f(t[j],y[j])*3 -f(t[j-1],y[j-1]))*(diff_n*0.5)
    return t,y

#################################################################
## Test Code ##
## You are highly encouraged to write your own tests as well
#################################################################

################
#%% Q3 Test
################

# Initialise
a = 0
b = 2
ya = 0.5
n = 40

# Define the ODE and the true solution
f = lambda t, y: y - t**2 + 1
y_true = lambda t: (t + 1)**2 - 0.5*np.exp(t)

# Compute the numerical solutions
t_euler, y_euler = adams_bashforth(f, a, b, ya, n, 1)
t_ab, y_ab = adams_bashforth(f, a, b, ya, n, 2)

print("\n################################")
print("Q3 TEST OUTPUT (last few values of solutions):\n")

# Print the last few points of each solution for comparison
print("  t      True      Euler    Adams-Bashforth")
print("--------------------------------------------")
for i in range(-4, 0):
    print("{:.2f}   {:.6f}   {:.6f}   {:.6f}".format(t_euler[i], 
          y_true(t_euler[i]), y_euler[i], y_ab[i]))

############################################################

