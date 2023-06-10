#################################################################
## Imports
## - No further imports should be necessary
#################################################################
import numpy as np

def adams_bashforth(f,a,b,ya,n,method):
  """
    Approximates the solution y(t) for an initial value problem, over the interval [a,b],
    with RHS function given in f and initial condition given in ya. Implements Euler's method
    if method=1 and 2-step Adams-Bashforth method if method=2.

    Parameters:
    a (float): The lower limit of integration.
    b (float): The upper limit of integration.
    n (int): The number of subintervals to use.
    f (function): The function to find the numerical solution for the first order ODE.
    ya (float): The initial condition/value for y.
    method (int): Method selector. 1 represents Euler's Method, 2 represents Adams-Bashforth method.

    Returns:
    t (ndarray): The position in the interval [a,b].
    y (ndarray): The point y(t) approximated based on method 1 or 2.
    """

    #assert
    if not isinstance(n, int) or n <= 0:
        raise ValueError("The input needs to be a positive integer.")
    #initual set
    y,t = np.zeros(n+1),np.linspace(a,b,n+1)
    h = (b-a)/n
    y[0]=ya
    #Euler = 1
    if method ==1:
        for i in range(n):
            y[i+1] = f(t[i],y[i])*h +y[i]
    #Adam-Bashforth =2
    elif method ==2:
        y[1] = y[0]+ f(t[0],y[0])*h # Initial approximation using Euler's Method
        for j in range(1,n):
            y[j+1]=  y[j]+  (f(t[j],y[j])*3 -f(t[j-1],y[j-1]))*(h*0.5)
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

