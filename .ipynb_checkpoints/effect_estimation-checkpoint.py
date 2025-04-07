import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (Lasso, LassoCV, LogisticRegression,
                                  LogisticRegressionCV,LinearRegression,
                                  MultiTaskElasticNet,MultiTaskElasticNetCV)
from sklearn.ensemble import (RandomForestRegressor,RandomForestClassifier,
                              GradientBoostingRegressor,GradientBoostingClassifier)
from econml.dml import DML, LinearDML, SparseLinearDML, CausalForestDML


class EstimationPipeline():

    def __init__(self, 
                 Y,
                 T, 
                 X, 
                 W,
                 estimator,
                 train_size: float = 0.7,
                 true_effect: float = None,
                 random_state: int = 123):

        # Get data splits.
        self.Y = Y
        self.T = T
        self.X = X
        self.W = W
        self.random_state = random_state
        self.split_data(train_size = train_size)

        # Store ground truth effect size, if available.
        self.true_effect = true_effect

        # Fit EconML DML estimator.
        self.estimator = estimator
        self.estimator.fit(self.Y_train, 
                           self.T_train, 
                           X = self.X_train,
                           W = self.W_train)


    def split_data(self, train_size: float = 0.7):

        '''
        Should split be stratified? Stratify on treatment.
        '''

        if self.X is None and self.W is None:
            Y_train, Y_vt, T_train, T_vt = train_test_split(self.Y,
                                                            self.T, 
                                                            test_size = 1-train_size,
                                                            random_state = self.random_state)
            Y_val, Y_test, T_val, T_test = train_test_split(Y_vt, 
                                                            T_vt, 
                                                            test_size = 0.5,
                                                            random_state = self.random_state)
        elif self.X is None:
            Y_train, Y_vt, T_train, T_vt, W_train, W_vt = train_test_split(self.Y,
                                                                           self.T, 
                                                                           self.W,
                                                                           test_size = 1-train_size,
                                                                           random_state = self.random_state)
            Y_val, Y_test, T_val, T_test, W_val, W_test = train_test_split(Y_vt, 
                                                                           T_vt,
                                                                           W_vt,
                                                                           test_size = 0.5,
                                                                           random_state = self.random_state)
        elif self.W is None:
            Y_train, Y_vt, T_train, T_vt, X_train, X_vt = train_test_split(self.Y,
                                                                           self.T, 
                                                                           self.X,
                                                                           test_size = 1-train_size,
                                                                           random_state = self.random_state)
            Y_val, Y_test, T_val, T_test, X_val, X_test = train_test_split(Y_vt, 
                                                                           T_vt,
                                                                           X_vt,
                                                                           test_size = 0.5,
                                                                           random_state = self.random_state)
        else:
            Y_train, Y_vt, T_train, T_vt, X_train, X_vt, W_train, W_vt = train_test_split(self.Y,
                                                                                          self.T, 
                                                                                          self.X, 
                                                                                          self.W, 
                                                                                          test_size = 1-train_size,
                                                                                          random_state = self.random_state)
            Y_val, Y_test, T_val, T_test, X_val, X_test, W_val, W_test = train_test_split(Y_vt, 
                                                                                          T_vt, 
                                                                                          X_vt, 
                                                                                          W_vt, 
                                                                                          test_size = 0.5,
                                                                                          random_state = self.random_state)
        self.Y_train = Y_train
        self.T_train = T_train
        self.Y_val = Y_val
        self.T_val = T_val
        self.Y_test = Y_test
        self.T_test = T_test

        if self.X is not None:
            self.X_train = X_train
            self.X_val = X_val
            self.X_test = X_test
        else:
            self.X_train, self.X_val, self.X_test = None, None, None

        if self.W is not None:
            self.W_train = W_train
            self.W_val = W_val
            self.W_test = W_test
        else:
            self.W_train, self.W_val, self.W_test = None, None, None

    

    
    def estimate_hte(self, verbose: bool = False):

        '''
        From the EconML documentation:
        Calculate the heterogeneous treatment effect tau(X, T0, T1).
        The effect is calculated between the two treatment points conditional 
        on a vector of features on a set of m test samples {T0_i, T1_i, X_i}.
        '''
        
        self.hte_train = self.estimator.effect(self.X_train)
        self.hte_val = self.estimator.effect(self.X_val)
        self.hte_test = self.estimator.effect(self.X_test)

        train_inf = self.estimator.effect_inference(self.X_train)
        val_inf = self.estimator.effect_inference(self.X_val)
        test_inf = self.estimator.effect_inference(self.X_test)
        
        self.df_hte_train = train_inf.summary_frame(alpha = 0.05, 
                                                    value = 0, 
                                                    decimals = 3)
        self.df_hte_val = val_inf.summary_frame(alpha = 0.05, 
                                                value = 0,  
                                                decimals = 3)
        self.df_hte_test = test_inf.summary_frame(alpha = 0.05, 
                                                  value = 0, 
                                                  decimals = 3)
        self.hte_pop_summary_train = train_inf.population_summary(alpha = 0.05, 
                                                                  value = 0,
                                                                  decimals = 3, 
                                                                  tol = 0.001)
        self.hte_pop_summary_val = val_inf.population_summary(alpha = 0.05, 
                                                              value = 0,
                                                              decimals = 3, 
                                                              tol = 0.001)
        self.hte_pop_summary_test = test_inf.population_summary(alpha = 0.05, 
                                                                value = 0,
                                                                decimals = 3, 
                                                                tol = 0.001)

        if verbose:
            self.print_hte_splits()


    def estimate_ate(self, verbose: bool = False):

        '''
        Inference results for the quantity E_X[tau(X, T0, T1)], i.e.,
        the average treatment effect.
        '''

        self.ate_train = self.estimator.ate(self.X_train)
        self.ate_val = self.estimator.ate(self.X_val)
        self.ate_test = self.estimator.ate(self.X_test)

        self.ate_summary_train = self.estimator.ate_inference(self.X_train)
        self.ate_summary_val = self.estimator.ate_inference(self.X_val)
        self.ate_summary_test = self.estimator.ate_inference(self.X_test)

        if verbose:
            display(self.ate_summary_test)


    def score(self, verbose: bool = False):

        '''
        From the EconML documentation:
        >Score the fitted CATE model on a new data set. Generates nuisance parameters
        for the new data set based on the fitted residual nuisance models created at fit time.
        It uses the mean prediction of the models fitted by the different crossfit folds.
        Then calculates the MSE of the final residual Y on residual T regression.

        If self.true_effect is not None, MSE is also computed for each data split.
        '''
        
        self.score_val = self.estimator.score(self.Y_val, self.T_val, self.X_val, self.W_val)
        self.score_test = self.estimator.score(self.Y_test, self.T_test, self.X_test, self.W_test)
        if verbose:
            print()
            print("Validation score :", round(self.score_val, 3))
            print("Test score       :", round(self.score_test, 3))
        
        if self.true_effect is not None:
            self.mse_train = ((self.true_effect - self.ate_train)**2).mean()
            self.mse_val = ((self.true_effect - self.ate_val)**2).mean()
            self.mse_test = ((self.true_effect - self.ate_test)**2).mean()
            if verbose:
                print("Train MSE        :", round(self.mse_train, 7))
                print("Validation MSE   :", round(self.mse_val, 7))
                print("Test MSE         :", round(self.mse_test, 7))
        

    def print_hte_splits(self):
        print()
        if isinstance(self.hte_pop_summary_train.mean_point, np.ndarray):
            print("Train split ATE : {} (CI [{}, {}])".format(round(self.hte_pop_summary_train.mean_point[0], 3), 
                                                              round(self.hte_pop_summary_train.conf_int_mean()[0][0], 3), 
                                                              round(self.hte_pop_summary_train.conf_int_mean()[1][0], 3)))
            print("Val split ATE   : {} (CI [{}, {}])".format(round(self.hte_pop_summary_val.mean_point[0], 3), 
                                                              round(self.hte_pop_summary_val.conf_int_mean()[0][0], 3), 
                                                              round(self.hte_pop_summary_val.conf_int_mean()[1][0], 3)))
            print("Test split ATE  : {} (CI [{}, {}])".format(round(self.hte_pop_summary_test.mean_point[0], 3), 
                                                              round(self.hte_pop_summary_test.conf_int_mean()[0][0], 3), 
                                                              round(self.hte_pop_summary_test.conf_int_mean()[1][0], 3)))
        else:
            print("Train split ATE : {} (CI [{}, {}])".format(round(self.hte_pop_summary_train.mean_point, 3), 
                                                              round(self.hte_pop_summary_train.conf_int_mean()[0], 3), 
                                                              round(self.hte_pop_summary_train.conf_int_mean()[1], 3)))
            print("Val split ATE   : {} (CI [{}, {}])".format(round(self.hte_pop_summary_val.mean_point, 3), 
                                                              round(self.hte_pop_summary_val.conf_int_mean()[0], 3), 
                                                              round(self.hte_pop_summary_val.conf_int_mean()[1], 3)))
            print("Test split ATE  : {} (CI [{}, {}])".format(round(self.hte_pop_summary_test.mean_point, 3), 
                                                              round(self.hte_pop_summary_test.conf_int_mean()[0], 3), 
                                                              round(self.hte_pop_summary_test.conf_int_mean()[1], 3)))


class Estimator():
    
    def get_wcde(self,
                 df, 
                 exposure, 
                 outcome, 
                 parents, 
                 linear_regression = True,
                 verbose = False):

        '''
        Estimate the WCDE using linear double ML or linear regression.
        Variables must be continuous.
        '''

        if linear_regression:
            wcde = self.get_CATE(df = df, 
                                 exposure = exposure, 
                                 outcome = outcome, 
                                 covars = parents, 
                                 model = "linear")
            return wcde
        else:
            Y = df[[outcome]].to_numpy().reshape(-1)
            T = df[[exposure]].to_numpy().reshape(-1)
            if len(parents) > 0:
                W = df[parents].to_numpy()
            else:
                W = None
            dml = LinearDML(model_y = "auto",
                            model_t = "auto",
                            discrete_outcome = False,
                            discrete_treatment = False,
                            random_state = 123)
            ep = EstimationPipeline(Y = Y,
                                    T = T,
                                    X = None,
                                    W = W, 
                                    estimator = dml,
                                    train_size = 0.7,
                                    true_effect = None)
        
            ep.estimate_hte(verbose = verbose)
            ep.score(verbose = verbose)
            if verbose:
                display(ep.estimator.summary(alpha = 0.05).tables[0])
        
            return ep.hte_val.item()


    def get_CATE(self,
                 df: pd.DataFrame, 
                 exposure: str, 
                 outcome: str, 
                 covars: list, 
                 model: str = "linear") -> float:
        
        if model == "linear":
            reg = LinearRegression().fit(df[[exposure] + covars], df[outcome])
        elif model == "logistic":
            reg = LogisticRegression(penalty = None, max_iter = 500).fit(df[[exposure] + covars], df[outcome])
        cate = reg.coef_[0]
        return cate
