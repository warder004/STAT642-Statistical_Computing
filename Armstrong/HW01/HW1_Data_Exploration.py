__author__ = 'drew'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import linear_model
from numpy import zeros, loadtxt
from pandas import Series, DataFrame


# predimension the array (totally cheating here)

X = zeros((5,1000,100))
Y = zeros((1000,5))

# Load the data files into the array

for i in range(5):
    X[i,:,:] = loadtxt('DataX{}.txt'.format(i))
    Y[:,i] = loadtxt('DataY{}.txt'.format(i))


# Save the correlation matricies as png files to be used in the LaTeX document

for i in range(5):
    X_df = DataFrame(X[i,:,:])
    corr = X_df.corr()
    
    print '\n' + 'Creating a Correlation Matrix Heat Map for X_{} and saving to PNG file X_{}_Corr.png'.format(i,i)
    fig=plt.figure()    
    plt.imshow(corr, cmap='hot', vmin=0, vmax=1)
    plt.colorbar()
    plt.xticks(range(0,len(corr)+1,10), range(0,len(corr)+1,10))
    plt.yticks(range(0,len(corr)+1,10), range(0,len(corr)+1,10));
    
    plt.savefig('X_{}_stan_Corr.png'.format(i), format='png')
    plt.close(fig)

for i in range(5):
    X_df = DataFrame(X[i,:,:])
    corr = X_df.corr()

    print '\n' + 'Creating a Correlation Matrix Heat Map for X_{} and saving to PNG file X_{}_Corr.png'.format(i,i)
    fig=plt.figure()
    plt.imshow(corr, cmap='hot')
    plt.colorbar()
    plt.xticks(range(0,len(corr)+1,10), range(0,len(corr)+1,10))
    plt.yticks(range(0,len(corr)+1,10), range(0,len(corr)+1,10));

    plt.savefig('X_{}_Corr.png'.format(i), format='png')
    plt.close(fig)