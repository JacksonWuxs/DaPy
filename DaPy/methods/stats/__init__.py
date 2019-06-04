from .variance_analysis import ANOVA
from .linear_regression import LinearRegression
from .discriminant_analysis import DiscriminantAnalysis
from .sign_test import SignTest, WicoxonTest, CoxStautTest, RunTest
from .compare_mean import IndependentTTest

__all__ = ['ANOVA', 'LinearRegression', 'DiscriminantAnalysis', 'SignTest']
