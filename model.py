import pandas as pd
import sklearn.metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import f_oneway

def selectkbest(X_train_scaled, y_train, n):
    '''
    selectkbest takes in X_train scaled, y_train, and a desired number of features and returns 
    the selected features to be used in modeling
    '''
    f_selector = SelectKBest(k=n)
    f_selector.fit(X_train_scaled, y_train)
    f_support = f_selector.get_support()
    f_feature = X_train_scaled.loc[:,f_support].columns.tolist()
    print(str(len(f_feature)), 'selected features')
    print(f_feature)
    return f_feature

def get_metrics(train, validate, test, x_col, y_col, y_pred, clf):
    '''
    get_metrics takes in a confusion matrix (cnf) for a binary classifier and prints out metrics based on
    values in variables named X_train, y_train, and y_pred.
    
    return: a classification report as a transposed DataFrame
    '''
    X_train, y_train = train[x_col], train[y_col]

    X_validate, y_validate = validate[x_col], validate[y_col]

    X_test, y_test = test[x_col], validate[y_col]
    
    accuracy = clf.score(X_train, y_train)
    class_report = pd.DataFrame(classification_report(y_train, y_pred, output_dict=True)).T
    conf = confusion_matrix(y_train, y_pred)
    tpr = conf[1][1] / conf[1].sum()
    fpr = conf[0][1] / conf[0].sum()
    tnr = conf[0][0] / conf[0].sum()
    fnr = conf[1][0] / conf[1].sum()
    print(f'''
    The accuracy for our model is {accuracy:.4}
    The True Positive Rate is {tpr:.3}, The False Positive Rate is {fpr:.3},
    The True Negative Rate is {tnr:.3}, and the False Negative Rate is {fnr:.3}
    ''')
    return class_report

def create_baseline(X_train, y_train):
    # 1. Create the object
    baseline = DummyClassifier(strategy='constant', constant=0)
    # 2. Fit the object
    baseline.fit(X_train, y_train)
    baseline_score = baseline.score(X_train, y_train)
    print(f'The baseline accuracy for hasWon in all cases on the League of Legends matches Dataset is {baseline_score:.4%}')
    return baseline_score
     
def decision_tree(X_train, X_validate, y_train, y_validate):
    '''
    Create model for decision tree use train and validate, return model and y_predict
    '''
    #set parameters, fit to our X and y train, get a score for train and validate sets.
    tree = DecisionTreeClassifier(max_leaf_nodes=12, max_depth=5)
    tree.fit(X_train,y_train)
    print(f'training score: {tree.score(X_train, y_train):.2%}')
    print(f'validate score: {tree.score(X_validate, y_validate):.2%}')
    y_pred = tree.predict(X_train)
    
    return tree, y_pred

def decision_tree_score(baseline, X_train, y_train, y_pred):
    '''
    This function just contain simple print statement for model score
    '''
    #decision tree scores
    tree_precision = round(sklearn.metrics.precision_score(y_train, y_pred, average='macro'),3)
    tree_recall = round(sklearn.metrics.recall_score(y_train, y_pred, average='macro'),3)
    print('Scores for Decision Tree!')
    print('---------------------------')
    print(f'Baseline score is {baseline:.3}')
    print(f'accuracy score is {round(sklearn.metrics.accuracy_score(y_train, y_pred),3)}')
    print(f'precision score is {tree_precision}')
    print(f'recall score is {tree_recall}')

def random_forest(X_train, X_validate, y_train, y_validate):
    '''
    Create model for random forest use train and validate, return model and y_predict
    '''
    #set parameters, fit to our X and y train, get a score for train and validate sets.
    rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=1,
                            n_estimators=100,
                            max_depth=3, 
                            random_state=123)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_train)
    print(f'training score: {rf.score(X_train, y_train):.2%}')
    print(f'validate score: {rf.score(X_validate, y_validate):.2%}')
    return rf, y_pred

def random_forest_score(baseline, X_train, y_train, y_pred):
    '''
    This function just contain simple print statement for model score
    '''
    print('Scores for Random Forest!')
    print('---------------------------')
    print(f'Baseline score is {baseline:.3}')
    print(f'accuracy score is {sklearn.metrics.accuracy_score(y_train, y_pred):.3}')
    print(f'precision score is {sklearn.metrics.precision_score(y_train, y_pred, pos_label =0,average="macro"):.3}')
    print(f'recall score is {sklearn.metrics.recall_score(y_train, y_pred, pos_label =0, average="macro"):.3}')

def kn_neigh(X_train, X_validate, y_train, y_validate):
    '''
    Create model for knn use train and validate, return model and y_predict
    '''
    #set parameters, fit to our X and y train, get a score for train and validate sets.
    knn = KNeighborsClassifier(n_neighbors = 15)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_train)
    print(f'training score: {knn.score(X_train, y_train):.2%}')
    print(f'validate score: {knn.score(X_validate, y_validate):.2%}')
    return knn, y_pred

def knn_score(baseline, X_train, y_train, y_pred):
    '''
    This function just contain simple print statement for model score
    '''
    print('Scores for KNeighbors!')
    print('---------------------------')
    print(f'Baseline score is {baseline:.3}')
    print(f'accuracy score is {sklearn.metrics.accuracy_score(y_train, y_pred):.3}')
    print(f'precision score is {sklearn.metrics.precision_score(y_train, y_pred, pos_label =0, average="macro"):.3}')
    print(f'recall score is {sklearn.metrics.recall_score(y_train, y_pred, pos_label =0, average="macro"):.3}')