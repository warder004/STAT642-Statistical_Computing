import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from numpy import zeros, loadtxt
from Regression_Class import reg_class

# Load the data files into the array

data_sets = list()
data_sets_small = list()

for i in range(5):
    data_sets.append(reg_class(i, loadtxt('DataX{}.txt'.format(i)), loadtxt('DataY{}.txt'.format(i))))

for dat in data_sets:
    data_sets_small.append(reg_class(dat.index, dat.x[0:100], dat.y[0:100]))

def hist_plot(hist, hist_x, title, num_bins = 50, color = 'red'):
     hist.hist(hist_x, num_bins, color = color)
     hist.set_xlabel('Residuals')
     hist.set_ylabel('Counts')
     hist.set_title(title)

depth = 20
bins = 50


print ''
print ''
print ''
print '-------------------------------------------------------'
print 'Starting process for small data sets'
print '-------------------------------------------------------'


print ''
print ''
print ''
print '-------------------------------------------------------'
print 'Regression Optimize at depth: {} and Histogram Plot'.format(depth)
print '-------------------------------------------------------'
for inst in data_sets_small:
    print 'Data set {}'.format(inst.index)

    inst.reg_fit(new_x=inst.x_train, new_y=inst.y_train)

    #                  reg_fit_resid(self, fit_x, true_y)
    resid_train = inst.reg_fit_resid(fit_x = inst.x_train, true_y = inst.y_train)
    resid_valid = inst.reg_fit_resid(fit_x = inst.x_valid, true_y = inst.y_valid)
    resid_test = inst.reg_fit_resid(fit_x = inst.x_test, true_y = inst.y_test)

    fig = plt.figure()
    train_plt = fig.add_subplot(1,3,1)
    hist_plot(train_plt, resid_train, 'Training', num_bins = bins, color = 'blue')
    valid_plt = fig.add_subplot(1,3,2)
    hist_plot(valid_plt, resid_valid, 'Validation', num_bins = bins, color = 'red')
    test_plt = fig.add_subplot(1,3,3)
    hist_plot(test_plt, resid_test, 'Testing', num_bins = bins, color = 'green')
    plt.tight_layout()
    plt.savefig('{}_reg_small_hist.png'.format(inst.index))
    plt.close('all')


print ''
print ''
print ''
print '-------------------------------------------------------'
print 'Lasso Optimize at depth: {} and Histogram Plot'.format(depth)
print '-------------------------------------------------------'
for inst in data_sets_small:
    print 'Data set {}'.format(inst.index)

    min_val = inst.las_find_best_alpha(depth=depth)
    print 'Lasso:  lowest lambda value is {} with an error of {} at index {}'.format(min_val[1], min_val[2], min_val[0])

    inst.las_fit(alpha=min_val[1], max_iter=1000, new_x=inst.x_train, new_y=inst.y_train)

    #                  reg_fit_resid(self, fit_x, true_y)
    resid_train = inst.las_fit_resid(fit_x = inst.x_train, true_y = inst.y_train)
    resid_valid = inst.las_fit_resid(fit_x = inst.x_valid, true_y = inst.y_valid)
    resid_test = inst.las_fit_resid(fit_x = inst.x_test, true_y = inst.y_test)

    fig = plt.figure()
    train_plt = fig.add_subplot(1,3,1)
    hist_plot(train_plt, resid_train, 'Training', num_bins = bins, color = 'blue')
    valid_plt = fig.add_subplot(1,3,2)
    hist_plot(valid_plt, resid_valid, 'Validation', num_bins = bins, color = 'red')
    test_plt = fig.add_subplot(1,3,3)
    hist_plot(test_plt, resid_test, 'Testing', num_bins = bins, color = 'green')
    plt.tight_layout()
    plt.savefig('{}_las_small_hist.png'.format(inst.index))
    plt.close('all')



print ''
print ''
print ''
print '-------------------------------------------------------'
print 'Ridge Optimize at depth: {} and Histogram Plot'.format(depth)
print '-------------------------------------------------------'
for inst in data_sets_small:
    print 'Data set {}'.format(inst.index)

    min_val = inst.rid_find_best_alpha(depth=depth)
    print 'Ridge:  lowest lambda value is {} with an error of {} at index {}'.format(min_val[1], min_val[2], min_val[0])

    inst.rid_fit(alpha=min_val[1], max_iter=1000, new_x=inst.x_train, new_y=inst.y_train)

    #                  reg_fit_resid(self, fit_x, true_y)
    resid_train = inst.rid_fit_resid(fit_x = inst.x_train, true_y = inst.y_train)
    resid_valid = inst.rid_fit_resid(fit_x = inst.x_valid, true_y = inst.y_valid)
    resid_test = inst.rid_fit_resid(fit_x = inst.x_test, true_y = inst.y_test)

    fig = plt.figure()
    train_plt = fig.add_subplot(1,3,1)
    hist_plot(train_plt, resid_train, 'Training', num_bins = bins, color = 'blue')
    valid_plt = fig.add_subplot(1,3,2)
    hist_plot(valid_plt, resid_valid, 'Validation', num_bins = bins, color = 'red')
    test_plt = fig.add_subplot(1,3,3)
    hist_plot(test_plt, resid_test, 'Testing', num_bins = bins, color = 'green')
    plt.tight_layout()
    plt.savefig('{}_rid_small_hist.png'.format(inst.index))
    plt.close('all')



print ''
print ''
print ''
print '-------------------------------------------------------'
print 'ElasticNet Optimize at depth: {} and Histogram Plot'.format(depth/2)
print '-------------------------------------------------------'
for inst in data_sets_small:
    print 'Data set {}'.format(inst.index)

    min_val = inst.eln_find_best_alpha_l1(depth=depth-12)
    print 'ElasticNet:  lowest alpha lambda pair is ({},{}) with an error of {} at index ({},{})'.format(min_val[1], min_val[3], min_val[4], min_val[0],  min_val[2])


    inst.eln_fit(alpha=min_val[1], l1_ratio=min_val[3], new_x=inst.x_train, new_y=inst.y_train)

    #                  reg_fit_resid(self, fit_x, true_y)
    resid_train = inst.eln_fit_resid(fit_x = inst.x_train, true_y = inst.y_train)
    resid_valid = inst.eln_fit_resid(fit_x = inst.x_valid, true_y = inst.y_valid)
    resid_test = inst.eln_fit_resid(fit_x = inst.x_test, true_y = inst.y_test)

    fig = plt.figure()
    train_plt = fig.add_subplot(1,3,1)
    hist_plot(train_plt, resid_train, 'Training', num_bins = bins, color = 'blue')
    valid_plt = fig.add_subplot(1,3,2)
    hist_plot(valid_plt, resid_valid, 'Validation', num_bins = bins, color = 'red')
    test_plt = fig.add_subplot(1,3,3)
    hist_plot(test_plt, resid_test, 'Testing', num_bins = bins, color = 'green')
    plt.tight_layout()
    plt.savefig('{}_eln_small_hist.png'.format(inst.index))
    plt.close('all')





print ''
print ''
print ''
print '-------------------------------------------------------'
print 'Starting process for large data sets'
print '-------------------------------------------------------'


print ''
print ''
print ''
print '-------------------------------------------------------'
print 'Regression Optimize at depth: {} and Histogram Plot'.format(depth)
print '-------------------------------------------------------'
for inst in data_sets:
    print 'Data set {}'.format(inst.index)

    inst.reg_fit(new_x=inst.x_train, new_y=inst.y_train)

    #                  reg_fit_resid(self, fit_x, true_y)
    resid_train = inst.reg_fit_resid(fit_x = inst.x_train, true_y = inst.y_train)
    resid_valid = inst.reg_fit_resid(fit_x = inst.x_valid, true_y = inst.y_valid)
    resid_test = inst.reg_fit_resid(fit_x = inst.x_test, true_y = inst.y_test)

    fig = plt.figure()
    train_plt = fig.add_subplot(1,3,1)
    hist_plot(train_plt, resid_train, 'Training', num_bins = bins, color = 'blue')
    valid_plt = fig.add_subplot(1,3,2)
    hist_plot(valid_plt, resid_valid, 'Validation', num_bins = bins, color = 'red')
    test_plt = fig.add_subplot(1,3,3)
    hist_plot(test_plt, resid_test, 'Testing', num_bins = bins, color = 'green')
    plt.tight_layout()
    plt.savefig('{}_reg_large_hist.png'.format(inst.index))
    plt.close('all')


print ''
print ''
print ''
print '-------------------------------------------------------'
print 'Lasso Optimize at depth: {} and Histogram Plot'.format(depth)
print '-------------------------------------------------------'
for inst in data_sets:
    print 'Data set {}'.format(inst.index)

    min_val = inst.las_find_best_alpha(depth=depth)
    print 'Lasso:  lowest lambda value is {} with an error of {} at index {}'.format(min_val[1], min_val[2], min_val[0])

    inst.las_fit(alpha=min_val[1], max_iter=1000, new_x=inst.x_train, new_y=inst.y_train)

    #                  reg_fit_resid(self, fit_x, true_y)
    resid_train = inst.las_fit_resid(fit_x = inst.x_train, true_y = inst.y_train)
    resid_valid = inst.las_fit_resid(fit_x = inst.x_valid, true_y = inst.y_valid)
    resid_test = inst.las_fit_resid(fit_x = inst.x_test, true_y = inst.y_test)

    fig = plt.figure()
    train_plt = fig.add_subplot(1,3,1)
    hist_plot(train_plt, resid_train, 'Training', num_bins = bins, color = 'blue')
    valid_plt = fig.add_subplot(1,3,2)
    hist_plot(valid_plt, resid_valid, 'Validation', num_bins = bins, color = 'red')
    test_plt = fig.add_subplot(1,3,3)
    hist_plot(test_plt, resid_test, 'Testing', num_bins = bins, color = 'green')
    plt.tight_layout()
    plt.savefig('{}_las_large_hist.png'.format(inst.index))
    plt.close('all')



print ''
print ''
print ''
print '-------------------------------------------------------'
print 'Ridge Optimize at depth: {} and Histogram Plot'.format(depth)
print '-------------------------------------------------------'
for inst in data_sets:
    print 'Data set {}'.format(inst.index)

    min_val = inst.rid_find_best_alpha(depth=depth)
    print 'Ridge:  lowest lambda value is {} with an error of {} at index {}'.format(min_val[1], min_val[2], min_val[0])

    inst.rid_fit(alpha=min_val[1], max_iter=1000, new_x=inst.x_train, new_y=inst.y_train)

    #                  reg_fit_resid(self, fit_x, true_y)
    resid_train = inst.rid_fit_resid(fit_x = inst.x_train, true_y = inst.y_train)
    resid_valid = inst.rid_fit_resid(fit_x = inst.x_valid, true_y = inst.y_valid)
    resid_test = inst.rid_fit_resid(fit_x = inst.x_test, true_y = inst.y_test)

    fig = plt.figure()
    train_plt = fig.add_subplot(1,3,1)
    hist_plot(train_plt, resid_train, 'Training', num_bins = bins, color = 'blue')
    valid_plt = fig.add_subplot(1,3,2)
    hist_plot(valid_plt, resid_valid, 'Validation', num_bins = bins, color = 'red')
    test_plt = fig.add_subplot(1,3,3)
    hist_plot(test_plt, resid_test, 'Testing', num_bins = bins, color = 'green')
    plt.tight_layout()
    plt.savefig('{}_rid_large_hist.png'.format(inst.index))
    plt.close('all')



print ''
print ''
print ''
print '-------------------------------------------------------'
print 'ElasticNet Optimize at depth: {} and Histogram Plot'.format(depth-8)
print '-------------------------------------------------------'
for inst in data_sets:
    print 'Data set {}'.format(inst.index)

    min_val = inst.eln_find_best_alpha_l1(depth=depth-12)
    print 'ElasticNet:  lowest alpha lambda pair is ({},{}) with an error of {} at index ({},{})'.format(min_val[1], min_val[3], min_val[4], min_val[0],  min_val[2])


    inst.eln_fit(alpha=min_val[1], l1_ratio=min_val[3], new_x=inst.x_train, new_y=inst.y_train)

    #                  reg_fit_resid(self, fit_x, true_y)
    resid_train = inst.eln_fit_resid(fit_x = inst.x_train, true_y = inst.y_train)
    resid_valid = inst.eln_fit_resid(fit_x = inst.x_valid, true_y = inst.y_valid)
    resid_test = inst.eln_fit_resid(fit_x = inst.x_test, true_y = inst.y_test)

    fig = plt.figure()
    train_plt = fig.add_subplot(1,3,1)
    hist_plot(train_plt, resid_train, 'Training', num_bins = bins, color = 'blue')
    valid_plt = fig.add_subplot(1,3,2)
    hist_plot(valid_plt, resid_valid, 'Validation', num_bins = bins, color = 'red')
    test_plt = fig.add_subplot(1,3,3)
    hist_plot(test_plt, resid_test, 'Testing', num_bins = bins, color = 'green')
    plt.tight_layout()
    plt.savefig('{}_eln_large_hist.png'.format(inst.index))
    plt.close('all')

