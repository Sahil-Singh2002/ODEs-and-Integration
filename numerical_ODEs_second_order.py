#################################################################
## Functions to carry out numerical solution of second order IVPS
#################################################################

#################################################################
## Imports
## - No further imports should be necessary
## - If you wish to import a non-standard modules, ask Ed if that 
## - is acceptable
#################################################################
import numpy as np

#################################################################
## Functions to be completed by student
#################################################################

#%% Q4a code
def adams_bashforth_2(f,a,b,alpha,beta,n,method):
    '''
    Complete the function so that it approximately solves 
    y'' = f(t,y,y') - y(a)=alpha,y'(a)=betaby applying either the Euler method 
    or the two-step method to y'=z,y(a)=alpha and z'=f(t,y,z),z(a)=beta 
    simultaneously.

    Parameters
    ----------
    a (float): The lower limit of integration
    b (float): The upper limit of integration
    n (int): The number of subintervals to use
    f (function): The function to find Numerical second order ODE for
    alpha (float): The point at which y(a) = alpha        
    beta (float): The point of y'(a)=z(a)= beta        
    method (int) : of only two types 1 representing Euler's Method and 
    2 representing Adam-Bashforth method.

    Returns
    -------
    t (ndarray): The position in the interval [a,b].
    y (ndarray): The point y(t) approximated based on method 1 or 2.
    '''
    t = np.linspace(a,b,n+1)
    z,z[0] = np.zeros(n+1),beta
    y,y[0] = np.zeros(n+1),alpha
    
    def f_system(t,Y):
        y = Y[0]
        z = Y[1]
        array = np.array([z, f(t,y,z)])
        return array
    
    diff_n = (b-a)/n
    #Euler = 1
    if method ==1:
        for i in range(0,n):
            z[i+1] = z[i]+f(t[i],y[i],z[i])*diff_n
            y[i+1] = y[i]+z[i]*diff_n
            
    #two setp adams.B method
    if method ==2:
        y[1] = y[0]+z[0]*diff_n
        z[1] = z[0]+f(t[0],y[0],z[0])*diff_n
        for j in range(1,n):
            #Adam.B two step formula
            z[j+1]= z[j]+ (3*f(t[j],y[j],z[j])-f(t[j-1],y[j-1],z[j-1]))*0.5*diff_n
            y[j+1]= y[j]+ (3*diff_n/2)*z[j] -(diff_n/2)*z[j-1]
    
    return t,y

#%% Q4b code
def compute_ode_errors(n_vals,no_n,a,b,alpha,beta,f,true_y):
    '''
    When looking at the date output from the function, we can see that the 
    Adam-Bashforth method results in the error to decrease rapidly while
    in comparison to Euler's Method as the step size is decreasing. This implies
    Adam Bashforth is more accurate then Euler's Method.
    '''
    
    errors_y = np.zeros((2,no_n))

    for i,n in enumerate(n_vals):
       for j in range(2):     
           errors_y[j,i] = abs( adams_bashforth_2(f,a,b,alpha,beta,n,j+1)[1][-1]
                            - true_y(b) )
    return errors_y

#################################################################
## Test Code ##
## You are highly encouraged to write your own tests as well,
## but these should be written in a separate file
#################################################################

# Define the second-order ODE
f = lambda t,y0,y1: (2 + np.exp(-t))*np.cos(t)-y0-2*y1
true_y = lambda t: np.exp(-t)- np.exp(-t)*np.cos(t) + np.sin(t)

a = 0
b = 1
alpha = 0
beta = 1

################
#%% Q4a Test
################

n = 40

# Compute the numerical solutions
t_euler, y_euler = adams_bashforth_2(f, a, b, alpha, beta, n, 1)
t_ab, y_ab = adams_bashforth_2(f, a, b, alpha, beta, n, 2)

print("\n################################")
print("Q4a TEST OUTPUT (last few values of solutions):\n")

# Print the last few points of each solution for comparison
print("  t      True      Euler    Adams-Bashforth")
print("--------------------------------------------")
for i in range(-4, 0):
    print("{:.2f}   {:.6f}   {:.6f}   {:.6f}".format(t_euler[i], 
          true_y(t_euler[i]), y_euler[i], y_ab[i]))
    

################
#%% Q4b Test
################

no_n = 6
n_vals = 4*2**np.arange(no_n)

errors_y = compute_ode_errors(n_vals,no_n,a,b,alpha,beta,f,true_y)

print("\n################################")
print("Q4b TEST OUTPUT:\n")

print("errors_y = \n")
print(errors_y)

############################################################

