import numpy as np
import matplotlib.pyplot as plt

# %% Functions from Project 1 Week 1 and 2
def infinity_norm(M):
    """
    Function for calculating an infinity norm
    _____________________________
    parameters:
    matrix : ndarray
        Some matrix of arbitrary dimensions

    return:
    infinity norm : float
        Value of the infinity norm
    """
    # We create a matrix with only the absolute values. Then we sum each of the rows, and find which row has the largest sum
    return max(np.sum(abs(M), axis = 1))

def condition_number(M):
    """
    Function for calculating an infinity bound
    _____________________________
    parameters:
    matrix : ndarray
        Some matrix of arbitrary dimensions

    return:
    infinity bound : float
        Value of the infinity bound
    """
    return infinity_norm(M) * infinity_norm(np.linalg.inv(M))

def LU_factorization(matrix):
    U = np.matrix.copy(matrix)
    n = len(U)   
    L = np.eye(n)  #Some identity matrix we need to transform

    #Pivot that exchanges rows if the diagonal element is zero
    for i in range(n):
        if U[i, i] == 0:
            for j in range(i+1, n):
                if U[i+1, j] != 0:
                    row_1 = np.copy(U[i])
                    row_2 = np.copy(U[j])
                    U[i] = row_2
                    U[j] = row_1
                    break

    for x in range(n-1):
        for i in range(x+1, n):
            L[i, x] = U[i, x]/U[x, x]
            U[i] = U[i] - U[i, x]/U[x, x]*U[x]
    return L, U

def forward_substitute(L, b):
    n = len(b)-1               
    x = np.zeros(len(b))       
    x[0] = b[0]/L[0, 0]        # The first x value can be found directly

    for i in range(1, n + 1):  # Now we loop forwards through the rows (skipping the first row cause we already fixed that one)
        x[i] = (1/L[i, i])*(b[i] -  sum([ L[i, j]*x[j] for j in range(0, n)])   )     #And we update according to the equation on page 65
    return x

def back_substitute(U, b):
    n = len(b)-1                  
    x = np.zeros(len(b))          
    x[n] = b[n]/U[n, n]           #The last x value can be found directly

    for i in range(n-1, -1, -1):  #Now we loop backwards from the last row (skipping the very last row cause we already fixed that one)
        x[i] = (1/U[i, i])*(b[i]- sum([U[i, j]*x[j]  for j in range(1, n+1)])     )   #And we update according to the buttom equation of page 65
    return x

def LU_solver(M, b):
    L, U = LU_factorization(M)
    y = forward_substitute(L, b)
    x = back_substitute(U, y)
    return x

def solve_alpha(w):
    new_matrix = E - w*S
    x = LU_solver(new_matrix, z)
    return np.dot(z, x)

def householder_QR_slow(A):
    M = np.copy(A)
    H_storage = []

    for i in range(np.shape(M)[1]):                                        #loop over columns in the matrix
        a = np.copy(M.T[i])
        a[:i] = 0                                                          #We only want to mess with everything below the diagonal
        alpha = -np.sign(a[i])*np.linalg.norm(a)                           #Calculate scalar with correct sign

        v = a - alpha * np.eye(np.shape(M)[0])[i]                          #Create the v vector
        H = np.eye(np.shape(M)[0]) - 2 * np.outer(v, v) / np.dot(v, v)     #Create the householder matrix

        H_storage.append(H)                                                #Store the householder matrix
        M = np.matmul(H, M)                                                #update the matrix 

    R = np.triu(M)                                                         #Extract the upper triangular part of the R matrix
    Q = np.transpose(np.linalg.multi_dot(H_storage))                       #Calculate Q by multiplying all the householder matrices together
    return Q, R

def householder_fast(A):
    M = np.copy(A)
    V = np.array([]).reshape(np.shape(M)[0], 0)                            #Create a empty matrix to store the v vectors
    
    for i in range(np.shape(M)[1]):                                        #loop over columns in the matrix
        a = np.copy(M.T[i])
        a[:i] = 0                                                          #We only want to mess with everything below the diagonal
        alpha = -np.sign(a[i])*np.linalg.norm(a)                           #Calculate scalar with correct sign

        v = a - alpha * np.eye(np.shape(M)[0])[i]                          #Create the v vector
        V = add_column(V, column = v)                                   #Store the v vector
        H = np.eye(np.shape(M)[0]) - 2 * np.outer(v, v) / np.dot(v, v)     #Create the householder matrix

        M = np.matmul(H, M)                                                #update the matrix 

    R = np.triu(M)                                                         #Extract the upper triangular part of the R matrix
    R = np.vstack([R, np.zeros(np.shape(R)[1])])                           #Add a row of zeros to the bottom of R   
    V = add_row(V, index = 0)                                           #Add a row of zeros to the top of v_storage
    VR = R + V                                                             #Add the two matrices together
    return VR

def VR_extract(VR):
    R = np.triu(VR)                                          #Extract the upper triangular part of the R matrix
    V = (VR-np.triu(VR))[1:]    
    R = R[:np.shape(R)[1]]                                        
    return R, V

def least_squares(A, b):
    a = np.copy(b)
    VR = householder_fast(A)                                   # We perform the householder factorization
    R, V = VR_extract(VR)                                      # We extract the R and V matrices               

    for v in V.T:                                              # We use the householder matrices on the vector b (but avoid the matrix and use the v)
        a = a - np.matmul(2*np.outer(v, a)/np.dot(v,v), v)
        
    c_1 = a[:len(R)]
    c_2 = a[len(R):]
    x = back_substitute(R, a[:len(R)])                      # We solve the system Rx = c_1
    r = np.linalg.norm(a[len(R):])**2
    return x, r


# %% Functions from project 2 week 3

def gershgorin(A):
    n = A.shape[0]
    centers = np.diag(A)
    radii = np.sum(abs(A), axis = 1) - abs(centers)
    return centers, radii

def plotter_gershgorin(centers, radii):
    fig, ax = plt.subplots(figsize = (6,3))
    ax.set_title("Gershgorin circles")
    ax.plot(np.real(centers), np.imag(centers), color = "red", marker = "x", linestyle = "", markersize = 15, label = "centers")
    ax.hlines(0, min(centers), max(centers), linestyle = "--", color = "black", label = "radii" )
    for x in range(len(centers)):
        circle_plotter(ax, np.real(centers[x]), np.imag(centers[x]), radii[x])

# %% Other self made functions


def random_matrix(N, seed = False, low = 0, high = 9):
    if seed:
        np.random.seed(seed)
    M = np.random.randint(low, high +1, (N, N)).astype(float)
    return M

def random_vector(N, seed = False, low = 0, high = 9):
    if seed:
        np.random.seed(seed)
    V = np.random.randint(low, high +1, N).astype(float)
    return V

def random_triangular_matrix(N, orientation = "upper", seed = False, low = 0, high = 9):
    M = random_matrix(N = N, seed = seed, low = low, high = high)

    if orientation == "lower" or orientation == 0:
        M = np.tril(M)
    if orientation == "upper" or orientation == 1:
        M = np.triu(M)
    return M

def is_same(A, B):
    return (A == B).all()

def is_singular(M):
    flag = False
    
    if np.linalg.det(M) == 0:
        flag = True
        return flag

    if np.linalg.matrix_rank(M) < np.shape(M)[0]:
        flag = True
        return flag

    return flag

def LU_solver(M, b):
    L, U = LU_factorization(M)
    y = forward_substitute(L, b)
    x = back_substitute(U, y)
    return x

def solve_alpha(E, S, z, w):

    new_matrix = E - w*S
    x = LU_solver(new_matrix, z)
    return np.dot(z, x)

def add_column(A = None, column = None, index = None):
    if A is None and column is None:
        raise ValueError("You need to input a column if the matrix is empty")
    
    if A is None:
        A = np.zeros((np.shape(column)[0], 0))

    if len(A) == 0 and column is None:
        raise ValueError("You need to input a column if the matrix is empty")
    
    if len(A) == 0:
        A = np.zeros((np.shape(column)[0], 0))

    if column is None:
        column = np.zeros(np.shape(A)[0])

    if index is not None:
        A = np.insert(A, index, column, axis = 1)
        return A
    
    return np.column_stack([A, column])

def add_row(A = None, row = None, index = None):
    if A is None and row is None:
        raise ValueError("You need to input a row if the matrix is empty")
    
    if A is None:
        A = np.zeros((0, np.shape(row)[0]))

    if len(A) == 0 and row is None:
        raise ValueError("You need to input a row if the matrix is empty")
    
    if len(A) == 0:
        A = np.zeros((0, np.shape(row)[0]))

    if row is None:
        row = np.zeros(np.shape(A)[1])

    if index is not None:
        A = np.insert(A, index, row, axis = 0)
        return A
    
    return np.vstack([A, row])

# %% Plot functions

def circle_plotter(ax, x, y, r):
    theta = np.linspace(0, 2*np.pi, 100)
    x1 = r*np.cos(theta) + x
    x2 = r*np.sin(theta) + y
    ax.plot(x1, x2, color = "red")

# %% Functions from project 3 week 4,5

def golden_section_min(f, a, b, tolerance = 1e-3, show_process = False):
    if show_process:
        storage = [[a, f(a)], [b, f(b)]]
    tau = (np.sqrt(5)-1)/2
    x_1 = a + (1-tau)*(b-a)
    x_2 = a + tau*(b-a)
    f_1 = f(x_1)
    f_2 = f(x_2)
    calls = 2
    while b-a > tolerance:
        if f_1 > f_2:
            a = x_1
            x_1 = x_2
            f_1 = f_2
            x_2 = a + tau*(b-a)
            f_2 = f(x_2)

        else:
            b = x_2
            x_2 = x_1
            f_2 = f_1
            x_1 = a + (1-tau)*(b-a)
            f_1 = f(x_1)

        calls += 1
        if show_process:
            storage.append([x_1, f_1])
            storage.append([x_2, f_2])

    if show_process:
        return (a+b)/2, calls, np.array(storage)
    else:
        return (a+b)/2, calls
    

# %% Functions from project 4 week 6

def plot_result(result, title, t = None, percentage = False):
    if t is None:
        t = np.linspace(0, len(result[0]), len(result[0]))

    final = np.round(result[:, -1], 2)
    plt.figure(figsize = (10, 5))  
    names = np.array(["Sick Homosexual Male ", "Sick Bisexual Male           ", "Sick Heterosexual Male    ", "Sick Heterosexual Female"])
    names = names + " - Final value: " + final.astype(str)


    plt.title(title)
    for x in range(len(result)):
        plt.plot(t, result[x], label = names[x])

    # Annotate final values in the plot (done with copilot)
    for i in range(len(result)):
        plt.annotate(str(np.round(result[i][-1], 2)), (t[-1], result[i][-1]), textcoords="offset points", xytext=(0,10), ha='center')
        
    plt.xlabel("Time")
    if percentage:
        plt.ylabel("Percentage of people infected")
    else:
        plt.ylabel("Number of people infected")
    plt.legend()
    plt.grid()







