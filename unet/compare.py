import torch

def calculate_relative_difference(matrix1, matrix2):
    difference = torch.abs(matrix1 - matrix2)
    relative_difference = difference / matrix2
    return relative_difference

def compare_matrices(source, calculation, target):
    """
    args:
        source: Source Matrix
        calculation: Calculated Matrix
        target: Target Matrix
    returns:
        difference between calculation and target in perspective to the source
        0: Worse than or equal to source
        0-1: Better than source
        1: Equal to traget
    """
    difference_source_target = calculate_relative_difference(source, target)
    difference_calculation_target = calculate_relative_difference(calculation, target)
    
    diff = 1 - torch.mean(difference_calculation_target).item() / torch.mean(difference_source_target).item()

    if diff >= 0:
        return diff
    else: 
        return 0
