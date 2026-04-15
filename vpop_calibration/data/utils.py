import pandas as pd


def join_if_two(tup: str, sep: str = "_") -> str:
    """Utility to concatenate a tuple of strings with a separator. Used to flatten dataframe index after pivotting."""
    if tup[0] == "":
        return tup[1]
    elif tup[1] == "":
        return tup[0]
    else:
        return sep.join(tup)


def create_tasks_maps(
    protocol_arms: list[str], outputs_names: list[str]
) -> tuple[list[str], dict[int, int], dict[int, str]]:
    """Util function to process a list of protocol arms and a list of outputs, into a set of tasks."""

    tasks_full_map: dict[str, tuple[str, str]] = {
        f"{output}_{protocol}": (output, protocol)
        for protocol in protocol_arms
        for output in outputs_names
    }
    tasks = list(tasks_full_map.keys())
    task_idx_to_output_idx: dict[int, int] = {
        tasks.index(task): outputs_names.index(output)
        for task, (output, _) in tasks_full_map.items()
    }
    task_idx_to_protocol: dict[int, str] = {
        tasks.index(task): protocol for task, (_, protocol) in tasks_full_map.items()
    }
    return tasks, task_idx_to_output_idx, task_idx_to_protocol


def normalize_dataframe(
    data_in: pd.DataFrame, ignore: list[str]
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Normalize a data frame with respect to its mean and std, ignoring certain columns, and output the corresponding mean and std."""
    selected_columns = data_in.columns.difference(ignore)
    norm_data = data_in
    mean = data_in[selected_columns].mean()
    std = data_in[selected_columns].std()
    norm_data[selected_columns] = (norm_data[selected_columns] - mean) / std
    return norm_data, mean, std
