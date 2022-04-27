import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats  
from sklearn.model_selection import train_test_split



def get_heatmap(df, target):
    '''returns a beautiful heatmap with correlations according to target variable'''
    positive_correlate = df.corr()[target].to_frame()
    heatmap = sns.heatmap(df.corr()[[target]].sort_values(by=target, ascending=False), vmin=-.5, vmax=.5, annot=True,cmap='seismic')
    heatmap.set_title(f'Features Correlated with {target}')
    return heatmap, positive_correlate

def bar_plot(df,target, attributes):
    #We obtain the number of rows that win and get the first towers  and we divide it by the total of rows to get the probability
    titles = ['Probability of winning of getting First Tower','Probability of winning of getting First Blood',
            'Probability of winning of getting Rift Herald', 'Probability of winning of getting Fire Dragon',
            'Probability of winning of getting Baron Nashor', 'Probability of winning of getting Elder Dragon' ]
    for title in titles:
        for atribute in attributes:
            Prob_winning = sum(np.logical_and(df[atribute] == 1, df[target] == 1)) / sum(df[atribute] == 1) * 100
            ax = sns.barplot(['Probability of winning', 'Probability of losing'], [Prob_winning, 100- Prob_winning])
            ax.set_ylim([0,100])
            ax.set_title(title)
        plt.plot()
        plt.show()

def lost_objectives(df, target, attributes):
    #We obtain the number of rows that win and lost objective and we divide it by the total of rows to get the probability
    for atribute in attributes:
        Prob_winning = sum(np.logical_and(df[atribute] == 1, df[target] == 1)) / sum(df[atribute] == 1) * 100
        ax = sns.barplot(['Probability of winning', 'Probability of losing'], [Prob_winning, 100- Prob_winning])
        ax.set_title(atribute)
        ax.set_ylim([0,100])
        plt.plot()
        plt.show()

def chi2test(df, x, y):
    '''
    performs a chi2 test for independence by taking in a dataframe, and 2 variables
    uses alpha of 0.05
    '''
    a = 0.05 #a for alpha 

    observed = pd.crosstab(df[x], df[y], margins = True)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    if p < a:
        print(f'Reject null hypothesis. There is evidence to suggest {x} and {y} are not independent.\n p-value is {p}')
    else:
        print(f"Fail to reject the null hypothesis. There is not sufficient evidence to reject independence of {x} and {y}.\n p-value is {p}")

