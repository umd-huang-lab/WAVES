import math


# Convert a list of tuples to a tuple of lists
def tuples_to_lists(data):
    # Ensure that all tuples in the list have the same length
    assert all(len(t) == len(data[0]) for t in data)
    # Unzips the list of tuples and converts the output to a tuple of lists
    return tuple(map(list, zip(*data)))


# Convert a tuple of lists to a list of tuples
def lists_to_tuples(data):
    # Ensure that all lists in the tuple have the same length
    assert all(len(lst) == len(data[0]) for lst in data)
    # Zips the tuple of lists and converts the output to a list of tuples
    return list(zip(*data))


# Get the order of magnitude of the smallest value in a list of values
def get_max_order_of_magnitude(values):
    max_order_of_magnitude = -math.inf
    for value in values:
        order_of_magnitude = math.floor(math.log10(value))
        if order_of_magnitude > max_order_of_magnitude:
            max_order_of_magnitude = order_of_magnitude
    return max_order_of_magnitude


# Format a mean and std into a string
def format_mean_and_std(mean, std, order_of_magnitude=None, style="latex"):
    assert style in ["ascii", "unicode", "latex"]
    if style == "ascii":
        pm_sign_func = lambda x, y: x + " +/- " + y
        exp_sign_func = lambda x: "* E" + str(x)
    elif style == "unicode":
        pm_sign_func = lambda x, y: x + " ± " + y
        exp_sign_func = lambda x: "x 10^" + str(x)
    else:
        pm_sign_func = lambda x, y: "$" + x + " \pm " + y + "$"
        exp_sign_func = lambda x: "$\\times 10^{" + str(x) + "}$"

    # Get the order of magnitude of the std
    if order_of_magnitude is None:
        order_of_magnitude = get_max_order_of_magnitude([std])

    # Format std to two significant figures in scientific notation
    std_fmt = "{:.1e}".format(std).split("e")
    coef_std, exp_std = std_fmt[0], int(std_fmt[1])

    # Adjust the mean's decimal point
    adjusted_mean = mean / (10**order_of_magnitude)
    coef_mean = f"{adjusted_mean:.1f}"

    return f"({pm_sign_func(coef_mean, coef_std)}) {exp_sign_func(exp_std)}"


# Format a list of means and stds into a list of strings
def format_mean_and_std_list(means, stds, order_of_magnitude=None, style="latex"):
    assert len(means) == len(stds)
    if order_of_magnitude is None:
        order_of_magnitude = get_max_order_of_magnitude(stds)
    fmt_strings = []
    for mean, std in zip(means, stds):
        fmt_strings.append(format_mean_and_std(mean, std, order_of_magnitude, style))
    return fmt_strings
