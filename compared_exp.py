import pandas as pd
from methods.normal import normal_test
from methods.uniform import uniform_test
from methods.gamma import gamma_test
from methods.lin import lin_test
from methods.drlb import *

# func = {'Normal':normal_test, 'Uniform':uniform_test, 'Gamma':gamma_test, 'Lin':lin_test, 'DRLB':DRLB_train}

def compared_exp(method_name, train_file_dict, test_file_dict, camp_n, budget_scaling):
    """
    This function takes in method_name, train_file_dict, test_file_dict, camp_n, and budget_scaling as parameters
    and returns a table containing the results of various tests based on the method_name.
    """
    total_budget = sum([train_file_dict[camp_id]["budget"] for camp_id in camp_n])
    total_impression = sum([train_file_dict[camp_id]["imp"] for camp_id in camp_n])
    table = pd.DataFrame(columns=['impressions', 'clicks', 'cost', 'win_rate', 'ecpc', 'ecpi', 'cer', 'wrc'])
    for camp_id in camp_n:
        budget = total_budget / total_impression * test_file_dict[camp_id]['imp'] * budget_scaling
        if method_name == 'Normal':
            res = normal_test(train_file_dict[camp_id], test_file_dict[camp_id], budget)
        if method_name == 'Uniform':
            res = uniform_test(train_file_dict[camp_id], test_file_dict[camp_id], budget)
        if method_name == 'Gamma':
            res = gamma_test(train_file_dict[camp_id], test_file_dict[camp_id], budget)
        if method_name == 'Lin':
            res = lin_test(train_file_dict[camp_id], test_file_dict[camp_id], budget)
        if method_name == 'DRLB':
            res = DRLB_train(train_file_dict[camp_id], test_file_dict[camp_id], budget)            

        # more elegant than the above
        # res = func[method_name](train_file_dict[camp_id], test_file_dict[camp_id], budget)
        
        table.loc[camp_id, :] = res
        
    return table