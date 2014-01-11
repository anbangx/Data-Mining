'''
Created on Jan 8, 2014

@author: anbangx
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def calculate_and_list_missing_data_percentage(data):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Calculate and list missing data percentage: ')
    missing_data_percentage = {}
    for j in range(data.shape[1]):
        count = 0
        for i in data.ix[:, j]:
            if str(i) == '?':
                count += 1
        missing_data_percentage[data.columns[j]] = count/(data.shape[0] * 1.0)
    print('--------------------------------------------------------------')
    print('The percentage of rows have missing values for each variable: ')
    print(missing_data_percentage)
    print('--------------------------------------------------------------')

def savefig(output_path, fig):
    fig.savefig(output_path)
    print('The result file is under ' + output_path)

def gen_missing_data_hist(data, save=True, show=False):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Generate missing data hist: ')
    missing_data_count = []
    for i in range(data.shape[0]):
        missing_data_count.append(0)
        for j in data.ix[i]:
            if j == '?':
                missing_data_count[i] += 1
    print(missing_data_count)

    xbins = [x for x in range(max(missing_data_count) + 1)]
    y = [0] * (max(missing_data_count) + 1)
    for x in xbins:
        y[x] = missing_data_count.count(x)

    fig = plt.figure()
    pos = np.arange(len(xbins))
    width = 1.0
    ax = plt.axes()
    ax.set_xticks(pos + (width / 2))
    ax.set_xticklabels(xbins)
    plt.bar(pos, y, width, color='b')
    if save:
        savefig('figure/missing_data_hist', fig)
    if show:
        plt.show()

def gen_num_of_unique_values(variables):
    num_unique_values_per_var = {}
    for j in range(variables.shape[1]):
        num_unique_values_per_var[variables.columns[j]] = np.size(np.unique(variables.ix[:, j]))
    return num_unique_values_per_var

def draw_hist(numeric_variables, column_name, bin=0, save=True, show=False):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Drawing hist for each numeric variable: ')
    fig = plt.figure()
    if bin > 0:
        numeric_variables[column_name].hist(bins=bin)
    else:
        numeric_variables[column_name].hist()
    if save:
        savefig('figure/' + column_name + '_hist', fig)
    if show:
        plt.show()

def draw_nonzero_hist(numeric_variables, column_name, save=True, show=False):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Drawing nonzero hist for each numeric variable: ')
    nonzero_column = numeric_variables[numeric_variables[column_name] != 0]
    if len(nonzero_column) == 0:
        return
    fig = plt.figure()
    nonzero_column[column_name].hist()
    if save:
        savefig('figure/' + column_name + '_nonzero_hist', fig)
    if show:
        plt.show()

def subplot(more_salary, less_salary, num_unique_value_per_var, save=True, show=False):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Subplotting for more salary and less salary: ')
    for j in range(less_salary.shape[1]):
        column_name = less_salary.columns[j]
        fig, axarr = plt.subplots(2, sharex=True)
        axarr[0].set_title(more_salary.columns[j] + '(>50K)')
        axarr[1].set_title(less_salary.columns[j] + '(<=50K)')
        if(num_unique_value_per_var[column_name] < 100):
            axarr[0].hist(more_salary.ix[:, j])
            axarr[1].hist(less_salary.ix[:, j])
        else:
            axarr[0].hist(more_salary.ix[:, j], bins=100)
            axarr[1].hist(less_salary.ix[:, j], bins=100)
        if save:
            savefig('figure/' + column_name + '_subplot', fig)
        if show:
            plt.show()

def boxplot(more_salary, less_salary, save=True, show=False):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Boxplotting for more salary and less salary: ')
    for j in range(less_salary.shape[1]):
        column_name = less_salary.columns[j]
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        more_salary.boxplot(column=column_name)
        plt.subplot(1, 2, 2)
        less_salary.boxplot(column=column_name)
        if save:
            savefig('figure/' + column_name + '_boxplot', fig)
        if show:
            plt.show()

def barplot_unique_categorical_values(categorial_variable, save=True, show=False):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Generating barplot for unique categorical values: ')
    for j in range(categorial_variable.shape[1]):
        column_name = categorial_variable.columns[j]
        grouped_by_age = categorial_variable.groupby(column_name)
        fig = plt.figure()
        grouped_by_age.size().plot(kind='bar')
        if save:
            savefig('figure/' + column_name + '_unique_values_barplot', fig)
        if show:
            plt.show()

def barplot_compare_two_classes(more_salary, less_salary, save=True, show=False):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Generating 2 barplots for each variable: ')
    for j in range(less_salary.shape[1]):
        column_name = less_salary.columns[j]

        fig = plt.figure()
        plt.subplot(2, 1, 1)
        grouped_by_age = more_salary.groupby(column_name)
        grouped_by_age.size().plot(kind='bar')

        plt.subplot(2, 1, 2)
        grouped_by_age = less_salary.groupby(column_name)
        grouped_by_age.size().plot(kind='bar')
        if save:
            savefig('figure/' + column_name + '_comparison_barplot', fig)
        if show:
            plt.show()

def entropy(f):
    ''' Takes a frequency and returns -1*f*log(f,2) '''
    return -1 * f * math.log(f, 2)

def compute_expected_information_gain(data):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Computing expected information gain: ')
    old_group = data[['salary']].groupby('salary')
    old_entropy = 0
    for k, v in old_group.groups.items():
        f = len(v) / (1.0 * data.shape[0])
        old_entropy += entropy(f)
    print('Old Entropy = ' + str(old_entropy))
    grouped_by_age = data[['age', 'salary']].groupby(['age'])
    expected_new_entropy = 0
    for name, group in grouped_by_age:
        size = len(group)
        for subname, subgroup in group.groupby(['salary']):
            subsize = len(subgroup)
            f = subsize / (1.0 * size)
            expected_new_entropy += entropy(f) * (size / (data.shape[0] * 1.0))
    print('Expected New Entropy = ' + str(expected_new_entropy))
    print('Expected information gain = ' + str(old_entropy - expected_new_entropy))

def compute_conditional_probabilities_for_age(data):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Computing conditional probabilities of education for age: ')
    binned_age = pd.cut(data.age, 2)
    data['binned_age'] = binned_age
    del data['age']
    for name, group in data[['binned_age', 'education']].groupby(['education']):
        ct = pd.crosstab(group['education'], group['binned_age'], margins=True,
                 rownames=['education'], colnames=['binned_age'])
        columns = group.columns
        for i in range(len(columns)):
            ct['P(age = ' + str(ct.columns[i]) + ' | education = ' + str(name) + ')'] = np.true_divide(
                ct[ct.columns[i]], ct['All'])
        print(ct)

def check_pairwise_dependency(data, variable1, variable2):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Checking pairwise dependency between ' + variable1 + ' and ' + variable2 + ': ')
    for name, group in data[[variable1, variable2]].groupby([variable2]):
        ct = pd.crosstab(group[variable2], group[variable1], margins=True,
                 rownames=[variable2], colnames=[variable1])
        for i in range(len(ct.columns) - 1):
            ct['P(' + variable1 + ' = ' + str(ct.columns[i]) + ' | ' + variable2 + ' = ' + str(name) + ')'] = \
                np.true_divide(ct[ct.columns[i]], ct['All'])
        print(ct)

if __name__ == '__main__':
    # part 1 import csv data into programming environment
    variable_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
                      'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                      'hours-per-week', 'native-country', 'salary']
    data = pd.read_csv('test.data', delimiter=',', skipinitialspace=True, names=variable_names)

    # part 2 Exploratory Data Analysis
    ''' 1. Variable Definitions '''
    ''' 2. Missing data '''
    ''' 2.1. For each variable calculate and list what percentage of rows have missing values for that variable '''
    calculate_and_list_missing_data_percentage(data)

    ''' 2.2. Generate a histogram indicating how many rows have 0, 1, 2, 3, .... missing values '''
    gen_missing_data_hist(data)
    
    ''' 3. Numeric Variables: '''
    numeric_variables = data._get_numeric_data()
    # print(numeric_variables)
    ''' 3.1 list the number of unique values for each variable '''
    num_unique_value_per_var = gen_num_of_unique_values(numeric_variables)
    for k, v in num_unique_value_per_var.items():
        if v < 100:
            ''' 3.2 for variables with less than 100 values, generate a histogram where each bin corresponds to one of
                the variable values '''
            draw_hist(numeric_variables, k)
        else:
            ''' 3.3 for variables with 100 or more values, generate a histogram using 100 bins (your software should be
                able to automatically figure out where to place the 100 bins) '''
            draw_hist(numeric_variables, k, 100)
    
    ''' 3.4 for the variables capital-gain,capital-loss, since they have a very large fraction of 0 values, just plot histograms
        of the non-zero values and list what fraction of values of the variable are equal to 0. '''
    draw_nonzero_hist(numeric_variables, 'capital-gain')
    draw_nonzero_hist(numeric_variables, 'capital-loss')
    
    ''' 3.5 For each variable, now plot 2 histograms as part of the same figure (using the same bins as before), one histogram
        directly above the other and with the same bins for each histogram, where the top histogram is for rows assigned to the
        class >50k, and the lower histogram is for the class <=50k '''
    numeric_variables['salary'] = data['salary']
    more_salary = numeric_variables[numeric_variables['salary'] == '>50K']
    less_salary = numeric_variables[numeric_variables['salary'] == '<=50K']
    del more_salary['salary']
    del less_salary['salary']
    subplot(more_salary, less_salary, num_unique_value_per_var)
        
    ''' 3.6 For each variable, generate a figure with 2 boxplots, side by side, where the left boxplot is for rows assigned to
        the class >50k, and the right box-plot is for the class <=50k '''
    boxplot(more_salary, less_salary)
    
    # 4. Categorial Variables
    categorial_variable = data.drop(numeric_variables.columns, axis=1)
    categorial_variable['salary'] = data['salary']
    # print(categorial_variable)
    ''' 4.1 generate a bar-plot, where the values for each bar correspond to the unique categorical values for each variable.
        Include the "?" symbol (indicating a missing value) as one of the possible values in your bar-plot. '''
    barplot_unique_categorical_values(categorial_variable)
    
    ''' 4.2 For each variable generate 2 barplots in a single figure, one above the other, where the top barplot is for rows
        assigned to the class >50k, and the lower barplot is for rows assigned to the class <=50k '''
    more_salary = categorial_variable[categorial_variable['salary'] == '>50K']
    less_salary = categorial_variable[categorial_variable['salary'] == '<=50K']
    del more_salary['salary']
    del less_salary['salary']
    barplot_compare_two_classes(more_salary, less_salary)

    ''' 4.3 Compute the expected information gain (base log2), relative to the class variable '''
    compute_expected_information_gain(data)

    # 5. Pairwise dependency on Age
    ''' 5.1. compute the conditional probabilities of a categorial variable '''
    compute_conditional_probabilities_for_age(data.copy())

    ''' 5.2. pick any 2 of the numeric variables and explore whether or not they depend on each other,i.e., are they
        independent or not? if they are dependent, explain how you determined this and what the dependence is. If you
        wish (but this is not necessary) you can discretize the 2 variables you select for this part of the assignment.
    '''
    # variable1: age, variable2: sex
    check_pairwise_dependency(data, 'age', 'sex')
    # variable1: age, variable2: hours-per-week
    check_pairwise_dependency(data, 'age', 'hours-per-week')

        
        