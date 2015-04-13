from numpy.random import uniform as runiform
from numpy import zeros

xdim = 100

cmatrix = zeros((xdim,xdim))
cblank = zeros((xdim,xdim))
chybrid = zeros((xdim,xdim))
i = 0

while i < xdim:
    j = i + 1
    cmatrix[i,i] = 1
    cblank[i,i] = 1
    chybrid[i,i] = 1
    while j < xdim:
        rho = runiform(0.9,1,1)
        cmatrix[i,j] = rho
        cmatrix[j,i] = rho
        
        if i < xdim/2:
            if j < xdim/2:
                chybrid[i,j]=rho
                chybrid[j,i]=rho
            
        j = j + 1
    i = i + 1

nummodels = 5

beta0 = zeros(nummodels)    
beta = zeros((xdim,nummodels))
betascale = 10

j = 0

while j < nummodels:
    i = 0
    beta0[j] = runiform(-betascale,betascale,1)
    while i < xdim:
        beta[i,j] = runiform(-betascale,betascale,1)
        i = i + 1
    j = j + 1

sparsity = 10

from random import shuffle

i = 2

while i < nummodels-1:
    beta[sparsity:,i] = 0
    ind = list(range(xdim))
    shuffle(ind)
    beta[:,i] = beta[ind,i]
    i = i + 1

beta[xdim/2 + sparsity:,4] = 0

ind = list(range(xdim))[50:]
shuffle(ind)    

beta[50:,4] = beta[ind,4]    
    
numsamp = 1000

from numpy.random import multivariate_normal as rnorm

mean = zeros(xdim)

X = zeros((nummodels,numsamp,xdim))

X[0,:,:] = rnorm(mean,cblank,numsamp)
X[1,:,:] = rnorm(mean,cmatrix,numsamp)
X[2,:,:] = rnorm(mean,cblank,numsamp)
X[3,:,:] = rnorm(mean,cmatrix,numsamp)
X[4,:,:] = rnorm(mean,chybrid,numsamp)

Y = zeros((numsamp,nummodels))

from numpy import dot
from numpy.random import randn
from numpy import ones

i = 0
sigma = [5,5,5,5,5]

while i < nummodels:
    Y[:,i] = dot(X[i,:,:],beta[:,i]) + sigma[i]*randn(numsamp) + beta0[i]*ones(numsamp)
    i = i + 1

# sim complete, shuffle and write the data

ind = list(range(nummodels))

shuffle(ind)

from numpy import savetxt

beta = beta[:,ind]
X = X[ind,:,:]
Y = Y[:,ind]

savetxt("BetaKey_StudentName.txt",beta)

savetxt("DataX0.txt",X[0,:,:])
savetxt("DataY0.txt",Y[:,0])

savetxt("DataX1.txt",X[1,:,:])
savetxt("DataY1.txt",Y[:,1])

savetxt("DataX2.txt",X[2,:,:])
savetxt("DataY2.txt",Y[:,2])

savetxt("DataX3.txt",X[3,:,:])
savetxt("DataY3.txt",Y[:,3])

savetxt("DataX4.txt",X[4,:,:])
savetxt("DataY4.txt",Y[:,4])
