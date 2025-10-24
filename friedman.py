import math
import numpy as np
import pandas as pd
from scipy.stats import f
import matplotlib.pyplot as plt
from scipy.stats import rankdata


def rank_maker(single_row, top_down):
    """
    This method takes as input a single array with numeric values
    It will change these values with the corresponding rank
    :param single_row: a single array with numeric values
    :param top_down: (Boolean) - False, if the best value is the lowest item in the list
                               - True, if the best value is the highest item in the list
    :return: A single array with ranks
    """
    if top_down:
        single_row = np.array(single_row) * -1
    r = rankdata(single_row, method='average')
    return r.tolist()


def friedmann_formula(means, k, N):
    """
    This method calculates the Friedman statistic
    :param means: A single array with all the mean ranks over all datasets
    :param k: (Int) number of methods
    :param N: (Int) number of datasets
    :return: The Friedman statistic
    """
    mean_sum = sum(x**2 for x in means)

    p1 = (12 * N) / (k * (k+1))
    p2 = mean_sum
    p3 = (k * (k + 1)**2) / 4

    return p1 * (p2 - p3)


def f_distribution_score(X, k, N):
    """
    This uses the Friedman score to derive a better statistic distributed according to the F-distribution
    :param X: (Float) The Friedman statistic
    :param k: (Int) number of methods
    :param N: (Int) number of datasets
    :return: The derived statistic according to the F-distribution
    """
    upper = (N - 1) * X
    lower = N * (k - 1) - X
    return upper / lower


def f_critical_value(alpha, k, N):
    """
    This method will look up the critical value for the F-ditribution with (k-1) and (k-1)(N-1) degrees of freedom
    :param alpha: (float) Significance level
    :param k: (Int) number of methods
    :param N: (Int) number of datasets
    :return: the corresponding critical value
    """
    df1 = k - 1
    df2 = (N - 1)*(k-1)
    return f.ppf(1 - alpha, df1, df2)


def critical_difference_calculator(alpha, k, N):
    """
    This method will calculate the critical difference that is used by the Nemenyi test
    :param alpha: (float) Significance level
    :param k: (Int) number of methods
    :param N: (Int) number of datasets
    :return: the corresponding critical difference
    """
    l1 = [0, 0, 1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164]
    l2 = [0, 0, 1.645,  2.052,  2.291,  2.459,  2.589, 2.693, 2.780, 2.855, 2.920]
    if alpha == 0.05:
        q = l1[k]
    else:
        q = l2[k]

    upper = k * (k + 1)
    lower = 6 * N

    return q * math.sqrt(upper/lower)


def nemenyi_test(critical_difference, means):
    """
    This method will perform the nemenyi test to see if a method performs significantly better than any other one.
    :param critical_difference: (float) the corresponding critical difference
    :param means: (List) a single array with all the mean ranks
    :return: (Dictionary) where every pair of methods and the difference is noted iff one method outperforms the other significantly
    """
    big_difference = []
    for i in range(0, len(means)-1):
        for j in range(i, len(means)):
            if abs(means[i] - means[j]) > critical_difference:
                big_difference.append({"first_method":i,
                                       "second_method":j,
                                       "difference":abs(means[i] - means[j])})
    return big_difference


def friedmann_and_nemenyi_test(matrix, col_names, k, N, aplha, top_down, criteria):
    """
    This method will perform the friedmann test and the nemenyi post hoc test to investigate if certain methods are in
    fact better performing than others
    :param matrix: (2x2 Array) is a 2-dimensional array that contains the methods on the x-axis and the datasets on the
                    y-axis. It has to contain numerical scores that can be used to compare methods
    :param col_names: (List) is a single array containing the names of all the methods
    :param k: (Int) number of methods
    :param N: (Int) number of datasets
    :param aplha: (float) significance level
    """

    rank_matrix = []
    for i in range(0,len(matrix)):
        rank_matrix.append(rank_maker(matrix[i], top_down))

    mean_r = []
    for j in range(0, k):
        sum = 0
        for i in range(0,len(matrix)):
            sum += rank_matrix[i][j]
        mean_r.append(sum/N)
    print(col_names)
    print(mean_r)

    bar_df = pd.DataFrame({
        'Method':col_names,
        'Value':mean_r
    })

    plt.figure(figsize=(16, 10))
    plt.bar(bar_df['Method'], bar_df['Value'])
    plt.xlabel('Method name')
    plt.ylabel('Average rank')
    plt.title(f"Mean rank for criteria '{criteria}' across all datasets")
    plt.show()

    friedman_score = friedmann_formula(mean_r, k, N)
    f_score = f_distribution_score(friedman_score, k, N)
    critical_value = f_critical_value(aplha, k, N)
    critical_difference = critical_difference_calculator(aplha, k, N)
    nemenyi_test_results = nemenyi_test(critical_difference, mean_r)

    print(f"The Friedmann statistic is equal to: {friedman_score:.3f}")
    print(f"The F-distribution statistic is equal to: {f_score:.3f}")
    print(f"The critical value for the F-dsitribution with an alpha of {aplha} is equal to: {critical_value:.3f}")
    print(f"The critical difference (CD) value is: {critical_difference:.4f}")
    if len(nemenyi_test_results) == 0:
        print(f"According to the nemenyi post hoc test with an alpha of {aplha}, no methods were significantly "
              f"outperfoming each other")
    else:
        print(f"The nemenyi post hoc test has found {len(nemenyi_test_results)} combinations were a  method "
              f"significantly outperformed another:")
        for combination in nemenyi_test_results:
            index_1 = int(combination['first_method'])
            index_2 = int(combination['second_method'])
            if mean_r[index_1] < mean_r[index_2]:
                print(f"Method {col_names[index_1]} significantly performs better than {col_names[index_2]} with a difference "
                      f"of {combination['difference']:.4f}")
            else:
                print(f"{col_names[index_2]} significantly performs better than {col_names[index_1]} with a difference "
                      f"of {combination['difference']:.4f}")


def read_summary_csv(filepath, target_columns, sheet_name):
    """
    This method will read the comprehensive_result csv file created by the other code
    It will take this fill and remove unwanted information
    It will transform it into multiple different matrices that can be used by the friedmann test algorithm
    :param filepath: (String) is the path to where the comprehensive result file is located
    :param hyper_param: (dic) This is a dictionary containing the method name, depth param, param_type and param_value
                        These values are used to select only the instances of a method with the correct
                        hyperparameters
    :param roc_d: (Int) represents the depth of the basic roc search method needs to use
    :param remove_d: (Int) represents the depth of the remove hull points method need to use
    :param target_columns: (List) is a sinle array with the names of the criteria we want to use to check if there
                            is a method outperforming another in this area
    :return: final_dic: (Dic) it returns a dictionary that has the following information of each target column:
                        - a matrix containting the values achieved by our code
                        - The indices of which method has which column in the matrix
                        - number of rows and collumns
    """
    df_full = pd.read_excel(filepath, sheet_name=sheet_name)
    final_dic = []
    for criteria in target_columns:
        mask_df = df_full[['method','dataset',criteria]]

        N = mask_df['dataset'].nunique()
        k = mask_df['method'].nunique()

        categories = mask_df['dataset'].astype('category').cat.categories
        dataset_indices_dic = dict(zip(categories, range(len(categories))))

        categories = mask_df['method'].astype('category').cat.categories
        method_indices_dic = dict(zip(categories, range(len(categories))))

        matrix = np.zeros((N, k))

        for i in range(0, len(mask_df)):
            row = mask_df.iloc[i]
            matrix[dataset_indices_dic[row['dataset']]][method_indices_dic[row['method']]] = row[criteria]

        dic = {'criteria': criteria,
               'matrix': matrix,
               'methods': categories,
               'row_amount': N,
               'col_amount': k}
        final_dic.append(dic)
    return final_dic


def read_practise_csv(filepath):
    """
    reads the practise csv
    :param filepath:
    :return:
    """
    df = pd.read_csv(filepath)
    df_matrix = []
    cols = []
    for row in df:
        cols.append(row)
    for i in range(0,len(df)):
        l = (df.iloc[i].tolist()[1:])
        l = [float(x) for x in l]
        df_matrix.append(l)
    cols = cols[1:]
    return df, df_matrix, cols, len(cols), len(df_matrix)


# path = "data.csv"
# df, df_matrix, cols, col_length, row_length = read_practise_csv(path)
# friedmann_and_nemenyi_test(df_matrix, cols, col_length, row_length, 0.05, True)

path = "consolidated_depth_analysis.xlsx"
sheet_name = "Depth 3"



columns_of_interest = ['depth_auc', 'depth_time']
matrices = read_summary_csv(path, columns_of_interest, sheet_name)

for mat in matrices:
    high_best = False
    if mat['criteria'] == 'depth_auc':
        high_best = True
    print(f"Performing the Friedmann test for the criteria: {mat['criteria']}")

    friedmann_and_nemenyi_test(mat['matrix'], mat['methods'], mat['col_amount'], mat['row_amount'], 0.05, high_best, mat['criteria'])

    print("\n ------------------- \n")
