#!/usr/bin/python
'''
Created on Jan 8, 2014
@author: anbangx
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import operator
from matplotlib.backends.backend_pdf import PdfPages

# pp = PdfPages('foo.pdf')

def compute_unique_values(data):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Computing unique values for each variable:')
    dict = {}
    for j in range(data.shape[1]):
        grouped = data.groupby(data.columns[j])
        value_counts = grouped.size()
        dict[data.columns[j]] = str(value_counts.count())
    print(dict.items())

def create_dict_categorical_or_numeric(data):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Indicating which variable is categorical or numeric:')
    dict = {}
    for j in range(data.shape[1]):
        if type(data.ix[0, j]) is np.int64:
            dict[data.columns[j]] = 'numeric'
        else:
            dict[data.columns[j]] = 'categorical'
    print(dict.items())

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
    print(missing_data_percentage.items())
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
    # print(missing_data_count)

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
    print(str(sorted(num_unique_values_per_var.items())))
    return num_unique_values_per_var

def draw_hist(numeric_variables, column_name, bin=0, save=True, show=False):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Drawing hist for each numeric variable: ')
    fig = plt.figure()
    fig.suptitle(column_name)
    if bin > 0:
        numeric_variables[column_name].hist(bins=bin)
    else:
        numeric_variables[column_name].hist()
    if save:
        if bin >= 100:
            savefig('figure/bigger100/' + column_name + '_hist', fig)
        else:
            savefig('figure/less100/' + column_name + '_hist', fig)
    if show:
        plt.show()

def draw_nonzero_hist(numeric_variables, column_name, bin, save=True, show=False):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Drawing nonzero hist for each numeric variable: ')
    nonzero_column = numeric_variables[numeric_variables[column_name] != 0]
    fraction = (len(numeric_variables) - len(nonzero_column)) / (1.0 * (len(numeric_variables)))
    print(str(column_name) + ' 0 fraction is: ' + str(fraction))
    if len(nonzero_column) == 0:
        return
    fig = plt.figure()
    fig.suptitle(column_name)
    nonzero_column[column_name].hist(bins=bin)
    if save:
        savefig('figure/non-zero/' + column_name + '_nonzero_hist', fig)
    if show:
        plt.show()

def subplot(more_salary, less_salary, num_unique_value_per_var, save=True, show=False):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Subplotting for more salary and less salary: ')

    for j in range(less_salary.shape[1]):
        column_name = less_salary.columns[j]
        if column_name == 'capital-gain' or column_name == 'capital-loss':
            more = more_salary[more_salary[column_name] != 0]
            less = less_salary[less_salary[column_name] != 0]
            fig, axarr = plt.subplots(2, sharex=True)
            axarr[0].set_title(more.columns[j] + '(classes>50K)')
            axarr[1].set_title(less.columns[j] + '(classes<=50K)')
            # if(num_unique_value_per_var[column_name] < 100):
            #     axarr[0].hist(more.ix[:, j], num_unique_value_per_var[column_name] - 1)
            #     axarr[1].hist(less.ix[:, j], num_unique_value_per_var[column_name] - 1)
            # else:
            axarr[0].hist(more.ix[:, j], bins=100)
            axarr[1].hist(less.ix[:, j], bins=100)
            if save:
                savefig('figure/subplot/' + column_name + '_subplot', fig)
                # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
                # fig.savefig(pp, format='pdf')
            if show:
                plt.show()
        else:
            fig, axarr = plt.subplots(2, sharex=True)
            axarr[0].set_title(more_salary.columns[j] + '(classes>50K)')
            axarr[1].set_title(less_salary.columns[j] + '(classes<=50K)')
            if(num_unique_value_per_var[column_name] < 100):
                axarr[0].hist(more_salary.ix[:, j], num_unique_value_per_var[column_name] - 1)
                axarr[1].hist(less_salary.ix[:, j], num_unique_value_per_var[column_name] - 1)
            else:
                axarr[0].hist(more_salary.ix[:, j], bins=100)
                axarr[1].hist(less_salary.ix[:, j], bins=100)
            if save:
                savefig('figure/subplot/' + column_name + '_subplot', fig)
                # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
                # fig.savefig(pp, format='pdf')
            if show:
                plt.show()

def boxplot(more_salary, less_salary, save=True, show=False):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Boxplotting for more salary and less salary: ')
    for j in range(less_salary.shape[1]):
        column_name = less_salary.columns[j]
        fig = plt.figure()
        boxPlotArray = [more_salary.ix[:, j], less_salary.ix[:, j]]
        plt.boxplot(boxPlotArray)
        plt.title('BoxPlot for ' + column_name + ' variable')
        plt.ylabel(column_name)
        plt.xticks([1, 2], ['Salary >50K', 'Salary <=50K'])
        # fig, axarr = plt.subplots(2, sharey=True) TODO how to sharey
        # fig, axarr = plt.subplots(2)
        # plt.subplot(1, 2, 1)
        # plt.xlabel(column_name + '(classes>=50K)')
        # # fig.boxplot(more_salary, column=column_name)
        # more_salary.boxplot(column=column_name)
        # plt.subplot(1, 2, 2)
        # plt.xlabel(column_name + '(classes<50K)')
        # less_salary.boxplot(column=column_name)
        if save:
            savefig('figure/boxplot/' + column_name + '_boxplot', fig)
        if show:
            plt.show()

def boxplot_nonzero(more_salary, less_salary, column_name, save=True, show=False):
    fig = plt.figure()
    more_salary = more_salary[more_salary[column_name] != 0]
    less_salary = less_salary[less_salary[column_name] != 0]
    boxPlotArray = [more_salary[column_name], less_salary[column_name]]
    plt.boxplot(boxPlotArray)
    plt.title('BoxPlot for ' + column_name + ' variable')
    plt.ylabel(column_name)
    plt.xticks([1, 2], ['Salary >50K', 'Salary <=50K'])
    # plt.subplot(1, 2, 1)
    # plt.xlabel(column_name + '(classes>=50K)')
    # more_salary = more_salary[more_salary[column_name] != 0]
    # more_salary.boxplot(column=column_name)
    # plt.subplot(1, 2, 2)
    # plt.xlabel(column_name + '(classes<50K)')
    # less_salary = less_salary[less_salary[column_name] != 0]
    # less_salary.boxplot(column=column_name)
    if save:
        savefig('figure/boxplot/' + column_name + '-non-zero' + '_boxplot', fig)
    if show:
        plt.show()

def barplot_unique_categorical_values(categorial_variable, save=True, show=False):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Generating barplot for unique categorical values: ')
    for j in range(categorial_variable.shape[1]):
        column_name = categorial_variable.columns[j]
        grouped_by_age = categorial_variable.groupby(column_name)
        fig = plt.figure()
        fig.suptitle(column_name)
        grouped_by_age.size().plot(kind='bar')
        if save:
            savefig('figure/barplot/' + column_name + '_unique_values_barplot', fig)
        if show:
            plt.show()

def barplot_compare_two_classes(more_salary, less_salary, save=True, show=False):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Generating 2 barplots for each variable: ')
    for j in range(less_salary.shape[1]):
        column_name = less_salary.columns[j]

        fig = plt.figure()
        plt.subplot(2, 1, 1)
        fig.suptitle(column_name + '(classes>=50K)[top] vs classes<50K[bottom]')
        grouped_by_age = more_salary.groupby(column_name)
        grouped_by_age.size().plot(kind='bar')

        plt.subplot(2, 1, 2)
        grouped_by_age = less_salary.groupby(column_name)
        grouped_by_age.size().plot(kind='bar')
        if save:
            savefig('figure/2barplot/' + column_name + '_comparison_barplot', fig)
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
    for j in range(data.shape[1]):
        if data.columns[j] == 'salary':
            continue
        grouped_by_age = data[[data.columns[j], 'salary']].groupby([data.columns[j]])
        expected_new_entropy = 0
        dict = {}
        for name, group in grouped_by_age:
            size = len(group)
            for subname, subgroup in group.groupby(['salary']):
                subsize = len(subgroup)
                f = subsize / (1.0 * size)
                expected_new_entropy += entropy(f) * (size / (data.shape[0] * 1.0))
        dict[data.columns[j]] = expected_new_entropy
        print(str(sorted(dict.items())))
        print('Expected information gain = ' + str(old_entropy - expected_new_entropy))
        print('-----------------------------------------------------------------------------------------------------------')

def compute_conditional_probabilities_for_age(data, compare_column):
    print('-----------------------------------------------------------------------------------------------------------')
    print('Computing conditional probabilities of education for age: ')
    binned_age = pd.cut(data.age, 10)
    data['binned_age'] = binned_age
    del data['age']
    for name, group in data[['binned_age', compare_column]].groupby([compare_column]):
        ct = pd.crosstab(group[compare_column], group['binned_age'], margins=True,
                 rownames=[compare_column], colnames=['binned_age'])
        columns = group.columns
        for i in range(5):  # TODO fix
            ct['P(age = ' + str(ct.columns[i]) + ' | ' + compare_column + ' = ' + str(name) + ')'] = np.true_divide(
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
    show_in_win = False
    if len(sys.argv) == 2:
        show_in_win = sys.argv[0]
    # part 1 import csv data into programming environment
    variable_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
                      'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                      'hours-per-week', 'native-country', 'salary']
    global_data = pd.read_csv('adult.data', delimiter=',', skipinitialspace=True, names=variable_names)

    # age_vs_hours = [data['capital-gain'], data['capital-loss']]
    # print(str(np.cov(age_vs_hours)))
    # print(str(np.corrcoef(age_vs_hours)))
    ''' 1.1 figure out how many unique values there are for each variable '''
    # compute_unique_values(data)
    ''' 1.2 create a vector that indicates which variables are categorical and which are numeric '''
    create_dict_categorical_or_numeric(global_data)

    # part 2 Exploratory Data Analysis
    ''' 1. Variable Definitions
    age: the age of the individual as reported by the individual at the time of the 1990 census, in integer units of years.
    workclass: a section of society dependent on physical labor, especially when compensated with an hourly wage
    fnlwgt: final weight(no clear definition)
    education: the highest level of education
    education-num: the amount of people receiving education
    marital-status: indicates whether the person is married
    occupation: a job or profession
    relationship: the way in which two or more people or things are connected
    race: a classification system used to categorize humans into large and distinct populations or groups
    sex: specialized into male and female varieties
    capital-gain: a profit that results from a disposition of a capital asset, such as stock, bond or real estate
    capital-loss: the difference between a lower selling price and a higher purchase price, resulting in a financial loss for the seller
    hours-per-week: the number of hours to work per week
    native-country: which country you consider to be your "home" or the country of your birth
    '''
    ''' 2. Missing data '''
    ''' 2.1. For each variable calculate and list what percentage of rows have missing values for that variable '''
    calculate_and_list_missing_data_percentage(global_data)

    ''' 2.2. Generate a histogram indicating how many rows have 0, 1, 2, 3, .... missing values '''
    gen_missing_data_hist(global_data, show=show_in_win)

    ''' 3. Numeric Variables: '''
    numeric_variables = global_data._get_numeric_data()
    ''' 3.1 list the number of unique values for each variable '''
    num_unique_value_per_var = gen_num_of_unique_values(numeric_variables)
    for k, v in num_unique_value_per_var.items():
        if v < 100:
            ''' 3.2 for variables with less than 100 values, generate a histogram where each bin corresponds to one of
                the variable values '''
            draw_hist(numeric_variables, k, v, show=show_in_win)
        else:
            ''' 3.3 for variables with 100 or more values, generate a histogram using 100 bins (your software should be
                able to automatically figure out where to place the 100 bins) '''
            draw_hist(numeric_variables, k, 100, show=show_in_win)
    #
    # ''' 3.4 for the variables capital-gain,capital-loss, since they have a very large fraction of 0 values, just plot histograms
    #     of the non-zero values and list what fraction of values of the variable are equal to 0. '''
    draw_nonzero_hist(numeric_variables, 'capital-gain', num_unique_value_per_var['capital-gain'], show=show_in_win)
    draw_nonzero_hist(numeric_variables, 'capital-loss', num_unique_value_per_var['capital-loss'], show=show_in_win)

    ''' 3.5 For each variable, now plot 2 histograms as part of the same figure (using the same bins as before), one histogram
        directly above the other and with the same bins for each histogram, where the top histogram is for rows assigned to the
        class >50k, and the lower histogram is for the class <=50k '''
    numeric_variables['salary'] = global_data['salary']
    more_salary = numeric_variables[numeric_variables['salary'] == '>50K']
    less_salary = numeric_variables[numeric_variables['salary'] == '<=50K']
    del more_salary['salary']
    del less_salary['salary']
    subplot(more_salary, less_salary, num_unique_value_per_var, show=show_in_win)

    ''' 3.6 For each variable, generate a figure with 2 boxplots, side by side, where the left boxplot is for rows assigned to
        the class >50k, and the right box-plot is for the class <=50k '''
    boxplot(more_salary, less_salary, show=show_in_win)
    boxplot_nonzero(more_salary, less_salary, 'capital-gain', show=show_in_win)
    boxplot_nonzero(more_salary, less_salary, 'capital-loss', show=show_in_win)

    # 4. Categorial Variables
    categorial_variable = global_data.drop(numeric_variables.columns, axis=1)
    categorial_variable['salary'] = global_data['salary']
    # print(categorial_variable)
    ''' 4.1 generate a bar-plot, where the values for each bar correspond to the unique categorical values for each variable.
        Include the "?" symbol (indicating a missing value) as one of the possible values in your bar-plot. '''
    barplot_unique_categorical_values(categorial_variable, show=show_in_win)

    ''' 4.2 For each variable generate 2 barplots in a single figure, one above the other, where the top barplot is for rows
        assigned to the class >50k, and the lower barplot is for rows assigned to the class <=50k '''
    more_salary = categorial_variable[categorial_variable['salary'] == '>50K']
    less_salary = categorial_variable[categorial_variable['salary'] == '<=50K']
    del more_salary['salary']
    del less_salary['salary']
    barplot_compare_two_classes(more_salary, less_salary, show=show_in_win)

    ''' 4.3 Compute the expected information gain (base log2), relative to the class variable '''
    compute_expected_information_gain(global_data.copy())

    # 5. Pairwise dependency on Age
    ''' 5.1. compute the conditional probabilities of a categorial variable '''
    compute_conditional_probabilities_for_age(global_data.copy(), 'education-num')

    ''' 5.2. pick any 2 of the numeric variables and explore whether or not they depend on each other,i.e., are they
        independent or not? if they are dependent, explain how you determined this and what the dependence is. If you
        wish (but this is not necessary) you can discretize the 2 variables you select for this part of the assignment.
    '''
    # variable1: age, variable2: sex
    check_pairwise_dependency(global_data, 'education-num', 'hours-per-week')
    # variable1: age, variable2: hours-per-week
    check_pairwise_dependency(global_data, 'age', 'hours-per-week')

    binned_hours = pd.cut(global_data['hours-per-week'], 4)
    global_data['hours-per-week'] = binned_hours
    binned_age = pd.cut(global_data.age, 10)
    global_data['binned_age'] = binned_age
    del global_data['age']
    for name, group in global_data[['binned_age', 'hours-per-week']].groupby(['hours-per-week']):
        ct = pd.crosstab(group['hours-per-week'], group['binned_age'], margins=True,
                 rownames=['hours-per-week'], colnames=['binned_age'])
        columns = group.columns
        for i in range(5):  # TODO fix
            ct['P(age = ' + str(ct.columns[i]) + ' | ' + 'hours-per-week' + ' = ' + str(name) + ')'] = np.true_divide(
                ct[ct.columns[i]], ct['All'])
        print(ct)

    # binned_hours = pd.cut(data['hours-per-week'], 4)
    # data['hours-per-week'] = binned_hours
    # binned_age = pd.cut(data.age, 10)
    # data['binned_age'] = binned_age
    # del data['age']
    # for name, group in data[['binned_age', 'sex']].groupby(['binned_age']):
    #     ct = pd.crosstab(group['binned_age'], group['sex'], margins=True,
    #              rownames=['binned_age'], colnames=['sex'])
    #     columns = group.columns
    #     for i in range(5):  # TODO fix
    #         ct['P(sex = ' + str(ct.columns[i]) + ' | ' + 'age' + ' = ' + str(name) + ')'] = np.true_divide(
    #             ct[ct.columns[i]], ct['All'])
    #     print(ct)

    ''' 5.3 compute the coveriance coeffiency '''

    # pp.close()