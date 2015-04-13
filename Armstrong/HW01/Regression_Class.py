__author__ = 'drew'

# import matplotlib
# import matplotlib.pyplot as plt
from sklearn import linear_model

import numpy as np


class reg_class:
    def __init__(self, index, x, y):
        self.index = index
        self.x = x
        self.y = y
        self.x_train = self.x[0:len(x)/2, :]
        self.y_train = self.y[0:len(y)/2]
        self.x_valid = self.x[len(x)/2:3*len(x)/4, :]
        self.y_valid = self.y[len(y)/2:3*len(y)/4]
        self.x_test = self.x[3*len(x)/4:len(x), :]
        self.y_test = self.y[3*len(y)/4:len(y)]
        self.lm_reg = linear_model.LinearRegression()
        self.lm_las = linear_model.Lasso()
        self.lm_rid = linear_model.Ridge()
        self.lm_eln = linear_model.ElasticNet()

    # Linear Regression Functions

    def reg_fit(self, new_x=None, new_y=None):
        if new_x is None or new_y is None:
            new_x = self.x_train
            new_y = self.y_train
        self.lm_reg = linear_model.LinearRegression()
        self.lm_reg.fit(new_x, new_y)
        return self.lm_reg

    def reg_fit_coef(self):
        return self.lm_reg.intercept_, self.lm_reg.coef_

    def reg_fit_resid(self, fit_x, true_y):
        resid = self.lm_reg.predict(fit_x)-true_y
        return resid

    def reg_fit_resid_sqrd(self, fit_x, true_y):
        resid = self.reg_fit_resid(fit_x, true_y)
        return resid**2

    def reg_fit_mean_sqrd_resid(self, fit_x, true_y):
        sqrd_resid = self.reg_fit_resid_sqrd(fit_x, true_y)
        return np.mean(sqrd_resid)

    # Lasso Regression Functions

    def las_fit(self, alpha=1.0, max_iter=1000, new_x=None, new_y=None):
        if new_x is None or new_y is None:
            new_x = self.x_train
            new_y = self.y_train
        self.lm_las = linear_model.Lasso(alpha=alpha, max_iter=max_iter)
        self.lm_las.fit(new_x, new_y)
        return self.lm_las

    def las_fit_coef(self):
        return self.lm_las.intercept_, self.lm_las.coef_

    def las_fit_resid(self, fit_x, true_y):
        resid = self.lm_las.predict(fit_x)-true_y
        return resid

    def las_fit_resid_sqrd(self, fit_x, true_y):
        resid = self.las_fit_resid(fit_x, true_y)
        return resid**2

    def las_fit_mean_sqrd_resid(self, fit_x, true_y):
        sqrd_resid = self.las_fit_resid_sqrd(fit_x, true_y)
        return np.mean(sqrd_resid)

    def las_find_best_alpha(self, depth=1, low=-5, high=10, steps=7, fit_x=None, fit_y=None, lam_x=None, lam_y=None, max_iter=5000):
        if fit_x is None or fit_y is None:
            fit_x = self.x_train
            fit_y = self.y_train
        if lam_x is None or lam_y is None:
            lam_x = self.x_valid
            lam_y = self.y_valid

        lbd = np.logspace(low,high,steps)
        error = np.zeros(steps)
        self.las_fit(alpha=lbd[0], max_iter=max_iter)
        min_val=[0,lbd[0],self.las_fit_mean_sqrd_resid(lam_x, lam_y)]
        for i in range(steps):
            self.las_fit(alpha=lbd[i], max_iter=max_iter)
            error[i] = self.las_fit_mean_sqrd_resid(lam_x, lam_y)
            if error[i] < min_val[2]:
                min_val = [i, lbd[i], error[i]]

        if depth > 1:
            if min_val[0] <= 0:
                if np.floor(np.log10(lbd[min_val[0]])-1) > -10:
                    min_val = self.las_find_best_alpha(depth = depth-1, low = np.floor(np.log10(lbd[min_val[0]])-1), high = np.ceil(np.log10(lbd[min_val[0]+1])), steps = steps, fit_x = fit_x, fit_y = fit_y, lam_x = lam_x, lam_y = lam_y, max_iter = max_iter)
                else:
                    min_val = [-1, 0, min_val[2]]
            elif min_val[0] >= steps-1:
                min_val = self.las_find_best_alpha(depth = depth-1, low = np.floor(np.log10(lbd[min_val[0]-1])), high = 1+np.ceil(np.log10(min_val[1])), steps = steps, fit_x = fit_x, fit_y = fit_y, lam_x = lam_x, lam_y = lam_y, max_iter = max_iter)
            else:
                min_val = self.las_find_best_alpha(depth = depth-1, low = np.floor(np.log10(lbd[min_val[0]-1])), high = np.ceil(np.log10(lbd[min_val[0]+1])), steps = steps, fit_x = fit_x, fit_y = fit_y, lam_x = lam_x, lam_y = lam_y, max_iter = max_iter)
        return min_val

    def las_mse_vs_alpha(self, low = 0, high = 10, steps = 100, fit_x = None, true_y = None, max_iter = 5000):
        if fit_x is None or true_y is None:
            fit_x = self.x_train
            true_y = self.y_train
        alpha = np.linspace(low, high, steps)
        mse = np.zeros(len(alpha))
        for i in range(steps):
            self.las_fit(alpha=alpha[i], max_iter=max_iter)
            mse[i] = self.las_fit_mean_sqrd_resid(fit_x = fit_x, true_y = true_y)
        return alpha, mse


    # Ridge Regression Functions

    def rid_fit(self, alpha=1.0, max_iter=None, new_x=None, new_y=None):
        if new_x is None or new_y is None:
            new_x = self.x_train
            new_y = self.y_train
        self.lm_rid = linear_model.Ridge(alpha=alpha, max_iter=max_iter)
        self.lm_rid.fit(new_x, new_y)
        return self.lm_rid

    def rid_fit_coef(self):
        return self.lm_rid.intercept_, self.lm_rid.coef_

    def rid_fit_resid(self, fit_x, true_y):
        resid = self.lm_rid.predict(fit_x)-true_y
        return resid

    def rid_fit_resid_sqrd(self, fit_x, true_y):
        resid = self.rid_fit_resid(fit_x, true_y)
        return resid**2

    def rid_fit_mean_sqrd_resid(self, fit_x, true_y):
        sqrd_resid = self.rid_fit_resid_sqrd(fit_x, true_y)
        return np.mean(sqrd_resid)

    def rid_find_best_alpha(self, depth=1, low=-5, high=10, steps=7, fit_x=None, fit_y=None, lam_x=None, lam_y=None, max_iter=5000):
        if fit_x is None or fit_y is None:
            fit_x = self.x_train
            fit_y = self.y_train
        if lam_x is None or lam_y is None:
            lam_x = self.x_valid
            lam_y = self.y_valid

        lbd = np.logspace(low,high,steps)
        error = np.zeros(steps)
        self.rid_fit(alpha=lbd[0], max_iter=max_iter)
        min_val=[0,lbd[0],self.rid_fit_mean_sqrd_resid(lam_x, lam_y)]
        for i in range(steps):
            self.rid_fit(alpha=lbd[i], max_iter=max_iter)
            error[i] = self.rid_fit_mean_sqrd_resid(lam_x, lam_y)
            if error[i] < min_val[2]:
                min_val = [i, lbd[i], error[i]]

        if depth > 1:
            if min_val[0] <= 0:
                if np.floor(np.log10(lbd[min_val[0]])-1) > -10:
                    min_val = self.rid_find_best_alpha(depth = depth-1, low = np.floor(np.log10(lbd[min_val[0]])-1), high = np.ceil(np.log10(lbd[min_val[0]+1])), steps = steps, fit_x = fit_x, fit_y = fit_y, lam_x = lam_x, lam_y = lam_y, max_iter = max_iter)
                else:
                    min_val = [-1, 0, min_val[2]]
            elif min_val[0] >= steps-1:
                min_val = self.rid_find_best_alpha(depth = depth-1, low = np.floor(np.log10(lbd[min_val[0]-1])), high = 1+np.ceil(np.log10(min_val[1])), steps = steps, fit_x = fit_x, fit_y = fit_y, lam_x = lam_x, lam_y = lam_y, max_iter = max_iter)
            else:
                min_val = self.rid_find_best_alpha(depth = depth-1, low = np.floor(np.log10(lbd[min_val[0]-1])), high = np.ceil(np.log10(lbd[min_val[0]+1])), steps = steps, fit_x = fit_x, fit_y = fit_y, lam_x = lam_x, lam_y = lam_y, max_iter = max_iter)
        return min_val

    def rid_mse_vs_alpha(self, low = 0, high = 10, steps = 100, fit_x = None, true_y = None, max_iter = 5000):
        if fit_x is None or true_y is None:
            fit_x = self.x_train
            true_y = self.y_train
        alpha = np.linspace(low, high, steps)
        mse = np.zeros(len(alpha))
        for i in range(steps):
            self.rid_fit(alpha=alpha[i], max_iter=max_iter)
            mse[i] = self.rid_fit_mean_sqrd_resid(fit_x = fit_x, true_y = true_y)
        return alpha, mse

    # Elastic Net Regression Functions

    def eln_fit(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, new_x=None, new_y=None):
        if new_x is None or new_y is None:
            new_x = self.x_train
            new_y = self.y_train
        self.lm_eln = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)
        self.lm_eln.fit(new_x, new_y)
        return self.lm_eln

    def eln_fit_coef(self):
        return self.lm_eln.intercept_, self.lm_eln.coef_

    def eln_fit_resid(self, fit_x, true_y):
        resid = self.lm_eln.predict(fit_x)-true_y
        return resid

    def eln_fit_resid_sqrd(self, fit_x, true_y):
        resid = self.eln_fit_resid(fit_x, true_y)
        return resid**2

    def eln_fit_mean_sqrd_resid(self, fit_x, true_y):
        sqrd_resid = self.eln_fit_resid_sqrd(fit_x, true_y)
        return np.mean(sqrd_resid)

    def eln_find_best_alpha_l1(self, depth=1, alph_low=-5, alph_high=10, alph_steps=7, l1_low=0, l1_high=1, l1_steps=7, fit_x=None, fit_y=None, lam_x=None, lam_y=None, max_iter=10000):
        if fit_x is None or fit_y is None:
            fit_x = self.x_train
            fit_y = self.y_train
        if lam_x is None or lam_y is None:
            lam_x = self.x_valid
            lam_y = self.y_valid

        lbd = np.logspace(alph_low, alph_high, alph_steps)
        alp = np.linspace(l1_low, l1_high, l1_steps)
        error = np.zeros((alph_steps, l1_steps))
        self.eln_fit(alpha=lbd[0], max_iter=max_iter)
        min_val=[0, lbd[0], 0, alp[0], self.eln_fit_mean_sqrd_resid(lam_x, lam_y)]
        for i in range(alph_steps):
            for j in range(l1_steps):
                self.eln_fit(alpha=lbd[i], l1_ratio = alp[j], max_iter=max_iter)
                error[i,j] = self.eln_fit_mean_sqrd_resid(lam_x, lam_y)
                if error[i, j] < min_val[4]:
                    min_val = [i, lbd[i], j, alp[j], error[i,j]]

        if depth > 1:
            new_alph_low = alph_low
            new_alph_high = alph_high
            new_alph_steps = alph_steps
            new_l1_low = l1_low
            new_l1_high = l1_high
            new_l1_steps = l1_steps
            if min_val[0] <= 0:
                new_alph_low = np.floor(np.log10(lbd[min_val[0]])-1)
                new_alph_high = np.ceil(np.log10(lbd[min_val[0]+1]))
            elif min_val[0] >= alph_steps-1:
                new_alph_low = np.floor(np.log10(lbd[min_val[0]-1]))
                new_alph_high = 1+np.ceil(np.log10(min_val[1]))
            else:
                new_alph_low = np.floor(np.log10(lbd[min_val[0]-1]))
                new_alph_high = np.ceil(np.log10(lbd[min_val[0]+1]))

            if min_val[2] <= 0:
                if min_val[2] <= 0:
                    new_l1_low = 0
                else:
                    new_l1_low = l1_low - (alp[1]-alp[0])
                new_l1_high = alp[min_val[2]+1]
            elif min_val[2] >= l1_steps-1:
                new_l1_low = alp[min_val[2]-1]
                if min_val[2] >= 1:
                    new_l1_high = 1
                else:
                    new_l1_high = l1_high + (alp[1]-alp[0])
            else:
                new_l1_low = alp[min_val[2]-1]
                new_l1_high = alp[min_val[2]+1]

            min_val = self.eln_find_best_alpha_l1(depth = depth-1, alph_low = new_alph_low, alph_high = new_alph_high, alph_steps = new_alph_steps, l1_low = new_l1_low, l1_high = new_l1_high, l1_steps = new_l1_steps, fit_x = fit_x, fit_y = fit_y, lam_x = lam_x, lam_y = lam_y, max_iter = max_iter)

        return min_val

    def eln_mse_vs_alpha(self, low = 0, high = 10, steps = 100, rho = .25, fit_x = None, true_y = None, max_iter = 5000):
        if fit_x is None or true_y is None:
            fit_x = self.x_train
            true_y = self.y_train
        alpha = np.linspace(low, high, steps)
        mse = np.zeros(len(alpha))
        for i in range(steps):
            self.eln_fit(alpha=alpha[i], l1_ratio=rho, max_iter = max_iter)
            mse[i] = self.eln_fit_mean_sqrd_resid(fit_x = fit_x, true_y = true_y)
        return alpha, mse