import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from numpy import zeros, loadtxt
from Regression_Class import reg_class

# Load the data files into the array

data_sets = list()
data_sets_small = list()

resid_long = zeros(1)
resid_short = zeros(1)

for i in range(5):
    data_sets.append(reg_class(i, loadtxt('DataX{}.txt'.format(i)), loadtxt('DataY{}.txt'.format(i))))

for dat in data_sets:
    data_sets_small.append(reg_class(dat.index, dat.x[0:100], dat.y[0:100]))


print ''
print ''
print '-------------------------------------------------------'
print 'Staring Lasso Plot Generation'
print '-------------------------------------------------------'
print ''
print ''

print ''
print ''
print '-------------------------------------------------------'
print 'Lasso MSE vs Alpha plot generation for small data sets'
print '-------------------------------------------------------'
for inst in data_sets_small:
    print 'Data set {}'.format(inst.index)
    alpha, mse_train = inst.las_mse_vs_alpha(low = 0, high = 5, steps = 100, fit_x = inst.x_train, true_y = inst.y_train)
    alpha, mse_valid = inst.las_mse_vs_alpha(low = 0, high = 5, steps = 100, fit_x = inst.x_valid, true_y = inst.y_valid)
    alpha, mse_test = inst.las_mse_vs_alpha(low = 0, high = 5, steps = 100, fit_x = inst.x_test, true_y = inst.y_test)
    #for i in range(10):
    #    print 'alpha={}, mse={}'.format(alpha[i], mse[i])
    plt.figure()
    plt.title('Lasso MSE vs Alpha Plot')
    plt.subplot(1,1,1)
    plt.plot(alpha, mse_train, color="blue", linestyle = "-", label="Train")
    plt.plot(alpha, mse_valid, color="red", linestyle = "-", label="Validate")
    plt.plot(alpha, mse_test, color="green", linestyle = "-", label="Test")
    plt.legend(loc='lower right')
    plt.xlim(0,5)
    plt.xticks(np.linspace(0,5,11))
    plt.xlabel('Alpha')
    plt.ylim(0,max(max(mse_train), max(mse_valid), max(mse_test)))
    plt.yticks(np.floor(np.linspace(0, max(max(mse_train), max(mse_valid), max(mse_test)), 11)))
    plt.ylabel('Mean Squared Error')
    plt.tight_layout()
    plt.savefig('{}_las_small_mse.png'.format(inst.index))
    plt.close('all')


print ''
print ''
print '-------------------------------------------------------'
print 'Lasso MSE vs Alpha plot generation for large data sets'
print '-------------------------------------------------------'
for inst in data_sets:
    print 'Data set {}'.format(inst.index)
    alpha, mse_train = inst.las_mse_vs_alpha(low = 0, high = 5, steps = 100, fit_x = inst.x_train, true_y = inst.y_train)
    alpha, mse_valid = inst.las_mse_vs_alpha(low = 0, high = 5, steps = 100, fit_x = inst.x_valid, true_y = inst.y_valid)
    alpha, mse_test = inst.las_mse_vs_alpha(low = 0, high = 5, steps = 100, fit_x = inst.x_test, true_y = inst.y_test)
    #for i in range(10):
    #    print 'alpha={}, mse={}'.format(alpha[i], mse[i])
    plt.figure()
    plt.title('Lasso MSE vs Alpha Plot')
    plt.subplot(1,1,1)
    plt.plot(alpha, mse_train, color="blue", linestyle = "-", label="Train")
    plt.plot(alpha, mse_valid, color="red", linestyle = "-", label="Validate")
    plt.plot(alpha, mse_test, color="green", linestyle = "-", label="Test")
    plt.legend(loc='lower right')
    plt.xlim(0,5)
    plt.xticks(np.linspace(0,5,11))
    plt.xlabel('Alpha')
    plt.ylim(0,max(max(mse_train), max(mse_valid), max(mse_test)))
    plt.yticks(np.floor(np.linspace(0, max(max(mse_train), max(mse_valid), max(mse_test)), 11)))
    plt.ylabel('Mean Squared Error')
    plt.tight_layout()
    plt.savefig('{}_las_large_mse.png'.format(inst.index))
    plt.close('all')


print ''
print ''
print '-------------------------------------------------------'
print 'Staring Ridge Plot Generation'
print '-------------------------------------------------------'
print ''
print ''

print ''
print ''
print '-------------------------------------------------------'
print 'Ridge MSE vs Alpha plot generation for small data sets'
print '-------------------------------------------------------'
for inst in data_sets_small:
    print 'Data set {}'.format(inst.index)
    alpha, mse_train = inst.rid_mse_vs_alpha(low = 0, high = 5, steps = 100, fit_x = inst.x_train, true_y = inst.y_train)
    alpha, mse_valid = inst.rid_mse_vs_alpha(low = 0, high = 5, steps = 100, fit_x = inst.x_valid, true_y = inst.y_valid)
    alpha, mse_test = inst.rid_mse_vs_alpha(low = 0, high = 5, steps = 100, fit_x = inst.x_test, true_y = inst.y_test)
    #for i in range(10):
    #    print 'alpha={}, mse={}'.format(alpha[i], mse[i])
    plt.figure()
    plt.title('Ridge MSE vs Alpha Plot')
    plt.subplot(1,1,1)
    plt.plot(alpha, mse_train, color="blue", linestyle = "-", label="Train")
    plt.plot(alpha, mse_valid, color="red", linestyle = "-", label="Validate")
    plt.plot(alpha, mse_test, color="green", linestyle = "-", label="Test")
    plt.legend(loc='lower right')
    plt.xlim(0,5)
    plt.xticks(np.linspace(0,5,11))
    plt.xlabel('Alpha')
    plt.ylim(0,max(max(mse_train), max(mse_valid), max(mse_test)))
    plt.yticks(np.floor(np.linspace(0, max(max(mse_train), max(mse_valid), max(mse_test)), 11)))
    plt.ylabel('Mean Squared Error')
    plt.tight_layout()
    plt.savefig('{}_rid_small_mse.png'.format(inst.index))
    plt.close('all')


print ''
print ''
print '-------------------------------------------------------'
print 'Ridge MSE vs Alpha plot generation for large data sets'
print '-------------------------------------------------------'
for inst in data_sets:
    print 'Data set {}'.format(inst.index)
    alpha, mse_train = inst.rid_mse_vs_alpha(low = 0, high = 5, steps = 100, fit_x = inst.x_train, true_y = inst.y_train)
    alpha, mse_valid = inst.rid_mse_vs_alpha(low = 0, high = 5, steps = 100, fit_x = inst.x_valid, true_y = inst.y_valid)
    alpha, mse_test = inst.rid_mse_vs_alpha(low = 0, high = 5, steps = 100, fit_x = inst.x_test, true_y = inst.y_test)
    #for i in range(10):
    #    print 'alpha={}, mse={}'.format(alpha[i], mse[i])
    plt.figure()
    plt.title('Ridge MSE vs Alpha Plot')
    plt.subplot(1,1,1)
    plt.plot(alpha, mse_train, color="blue", linestyle = "-", label="Train")
    plt.plot(alpha, mse_valid, color="red", linestyle = "-", label="Validate")
    plt.plot(alpha, mse_test, color="green", linestyle = "-", label="Test")
    plt.legend(loc='lower right')
    plt.xlim(0,5)
    plt.xticks(np.linspace(0,5,11))
    plt.xlabel('Alpha')
    plt.ylim(0,max(max(mse_train), max(mse_valid), max(mse_test)))
    plt.yticks(np.floor(np.linspace(0, max(max(mse_train), max(mse_valid), max(mse_test)), 11)))
    plt.ylabel('Mean Squared Error')
    plt.tight_layout()
    plt.savefig('{}_rid_large_mse.png'.format(inst.index))
    plt.close('all')



print ''
print ''
print '-------------------------------------------------------'
print 'Staring ElasticNet Plot Generation'
print '-------------------------------------------------------'
print ''
print ''

print ''
print ''
print '-------------------------------------------------------'
print 'ElasticNet MSE vs Alpha plot generation for small data sets'
print '-------------------------------------------------------'
for inst in data_sets_small:
    for r in [.25, .5, .75]:
        print 'Data set {}, rho = {}'.format(inst.index, r)
        #                       eln_mse_vs_alpha(low = 0, high = 10, steps = 100, rho = .25, fit_x = None, true_y = None, max_iter = 5000):
        alpha, mse_train = inst.eln_mse_vs_alpha(low = 0, high = 5, steps = 100, rho = r, fit_x = inst.x_train, true_y = inst.y_train)
        alpha, mse_valid = inst.eln_mse_vs_alpha(low = 0, high = 5, steps = 100, rho = r,  fit_x = inst.x_valid, true_y = inst.y_valid)
        alpha, mse_test = inst.eln_mse_vs_alpha(low = 0, high = 5, steps = 100, rho = r,  fit_x = inst.x_test, true_y = inst.y_test)
        #for i in range(10):
        #    print 'alpha={}, mse={}'.format(alpha[i], mse[i])
        plt.figure()
        plt.title('ElasticNet MSE vs Alpha Plot for rho = {}'.format(r))
        plt.subplot(1,1,1)
        plt.plot(alpha, mse_train, color="blue", linestyle = "-", label="Train")
        plt.plot(alpha, mse_valid, color="red", linestyle = "-", label="Validate")
        plt.plot(alpha, mse_test, color="green", linestyle = "-", label="Test")
        plt.legend(loc='lower right')
        plt.xlim(0,5)
        plt.xticks(np.linspace(0,5,11))
        plt.xlabel('Alpha')
        plt.ylim(0,max(max(mse_train), max(mse_valid), max(mse_test)))
        plt.yticks(np.floor(np.linspace(0, max(max(mse_train), max(mse_valid), max(mse_test)), 11)))
        plt.ylabel('Mean Squared Error')
        plt.tight_layout()
        plt.savefig('{}_rho_small_mse_{}.png'.format(inst.index,int(r*100)))
        plt.close('all')


print ''
print ''
print '-------------------------------------------------------'
print 'Ridge MSE vs Alpha plot generation for large data sets'
print '-------------------------------------------------------'
for inst in data_sets:
    for r in [.25, .5, .75]:
        print 'Data set {}, rho = {}'.format(inst.index, r)
        #                       eln_mse_vs_alpha(low = 0, high = 10, steps = 100, rho = .25, fit_x = None, true_y = None, max_iter = 5000):
        alpha, mse_train = inst.eln_mse_vs_alpha(low = 0, high = 5, steps = 100, rho = r, fit_x = inst.x_train, true_y = inst.y_train)
        alpha, mse_valid = inst.eln_mse_vs_alpha(low = 0, high = 5, steps = 100, rho = r,  fit_x = inst.x_valid, true_y = inst.y_valid)
        alpha, mse_test = inst.eln_mse_vs_alpha(low = 0, high = 5, steps = 100, rho = r,  fit_x = inst.x_test, true_y = inst.y_test)
        #for i in range(10):
        #    print 'alpha={}, mse={}'.format(alpha[i], mse[i])
        plt.figure()
        plt.title('ElasticNet MSE vs Alpha Plot for rho = {}'.format(r))
        plt.subplot(1,1,1)
        plt.plot(alpha, mse_train, color="blue", linestyle = "-", label="Train")
        plt.plot(alpha, mse_valid, color="red", linestyle = "-", label="Validate")
        plt.plot(alpha, mse_test, color="green", linestyle = "-", label="Test")
        plt.legend(loc='lower right')
        plt.xlim(0,5)
        plt.xticks(np.linspace(0,5,11))
        plt.xlabel('Alpha')
        plt.ylim(0,max(max(mse_train), max(mse_valid), max(mse_test)))
        plt.yticks(np.floor(np.linspace(0, max(max(mse_train), max(mse_valid), max(mse_test)), 11)))
        plt.ylabel('Mean Squared Error')
        plt.tight_layout()
        plt.savefig('{}_rho_large_mse_{}.png'.format(inst.index,int(r*100)))
        plt.close('all')