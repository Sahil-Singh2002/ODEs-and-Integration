#################################################################
## Functions to carry out numerical integration
#################################################################

#################################################################
## Imports
## - No further imports should be necessary
#################################################################
import numpy as np
import matplotlib.pyplot as plt

#################################################################
## Functions to be completed by student
#################################################################

#%% Q1 code
def composite_trapezium(a,b,n,f):
    """
    To approximate the definite integral of the function f(x) over the interval [a,b]
    using the composite trapezium rule with n subintervals.   

    Parameters:
    -----------
    a (float): The lower limit of integration
    b (float): The upper limit of integration
    n (int): The number of subintervals to use in the composite trapezoidal rule
    f (function): The function to integrate
    
    Returns
    -------
    integral_approx (float): The approximation of the integral 

    Examples
    -------
    >>> integral_approx = compute_trapezium(0,1,10,lambda x: x**2)
    """
    
    x = np.linspace(a,b,n+1) #Construct the quadrature points
    h = (b-a)/n

    #Construct the quadrature weights: 
    #These are the coefficient w_i of f(x_i) in the summation
    weights = h*np.ones(n+1) 
    weights[[0,-1]] = h/2

    integral_approx = np.sum(f(x)*weights)

    return integral_approx


#%% Q2a code
def romberg_integration(a,b,n,f,level):
    '''
    To approximate the definite integral of the function f(x) over the interval [a,b]
    using the Richardson extrapolation to CTR(n) with the specified level of extrapolation (level).

    Parameters
    ----------
    a (float): The lower limit of integration
    b (float): The upper limit of integration
    n (int): The number of subintervals to use in the composite trapezoidal rule
    f (function): The function to integrate
    level (int): The stage of which the Romberg integral runs up to. 

    Returns
    -------
    integral_approx (float): This gives the approximated integral of the
    function f(x) in the interval [a,b] for Romberg integration approximation based on
    CTR(level*n). Giving R_(level,level).

    Examples
    -------
    >>> integral_approx = compute_trapezium(0,np.pi,2,lambda x: np.sin(x) )
    
    '''
    R = np.zeros((level+1,level+1))
    for i in range(0,level+1): 
        R[i,0]=composite_trapezium(a,b,n,f)
        n = 2*n
        for j in range(1,i+1):
            R[i,j] = (R[i,j-1]*4**(j) -R[i-1,j-1])/(4**(j) -1)

    integral_approx = R[level-1,level-1]
    return integral_approx

#%% Q2b code
def compute_errors(N,no_n,levels,no_levels,f,a,b,true_val):
    '''
    The true value of the integral of the function f(x) = 1/x is log(2)
    in the interval of [1,2]. We see a decrease in error. However for level 3 and 4
    we examin that the convergence is no longer uniform. This can be due to
    the singularity at point x = 0 for the function. Which clearly shows the 
    rate of convergence is being affected.
    '''
    fig, ax = plt.subplots()
    #Algorithm
    error_matrix = np.zeros((no_levels, no_n))
    for i,level in enumerate(levels):
        for j,num in enumerate(N):
            approx = romberg_integration(a,b,num,f,level) 
            error_matrix[i,j] = abs(approx -true_val)
        #Plot
        ax.loglog(N,error_matrix[i],
                  "o",label=f"Level {level}",linestyle ='-')  
        #Label for axies
        ax.set_xlabel("Number of subintervals n")
        ax.set_ylabel("Error")
        #Label for Title
        ax.set_title("Error vs # subintervals for levels")
        ax.legend(loc=(1.02, 0))
        
    plt.show()
    return error_matrix,fig
#################################################################
## Test Code ##
## You are highly encouraged to write your own tests as well
#################################################################

################
#%% Q1 Test
################

# Initialise
f = lambda x: np.sin(x)
a = 0
b = np.pi
n = 10

#Run the function
integral_approx = composite_trapezium(a,b,n,f)

print("\n################################")
print("Q1 TEST OUTPUT:\n")
print("integral_approx =\n")
print(integral_approx)
print("")

# Initialise
f = lambda x: np.sin(x)
a = 0
b = np.pi
n = 2

print("\n################################")
print("Q2a TEST OUTPUT:\n")
print("")

# Test code
f = lambda x: np.sin(x)
a = 0
b = np.pi
n = 2

for level in range(1,5):
    #Run test 
    integral_approx = romberg_integration(a,b,n,f,level)
    print("level = " + str(level)+ ", integral_approx = " + str(integral_approx))


N = [1,2,4,8]
levels = [1,2,3,4]
true_val = 2.0
error_matrix, fig = compute_errors(N,4,levels,4,f,a,b,true_val)

print("\n################################")
print("Q2b TEST OUTPUT:\n")
print("Error =\n")
print(error_matrix)

