import pandas as pd
import pandera.pandas as pa


def join_if_two(tup: str, sep: str = "_") -> str:
    """Utility to concatenate a tuple of strings with a separator. Used to flatten dataframe index after pivotting."""
    if tup[0] == "":
        return tup[1]
    elif tup[1] == "":
        return tup[0]
    else:
        return sep.join(tup)


class TaskMap:
    def __init__(self, protocol_arms: list[str], output_names: list[str]):
        """Utility class to process a list of protocol arms and a list of outputs, into a set of tasks."""
        self.protocol_arms = protocol_arms
        self.output_names = output_names
        self.full_tasks_map: dict[str, tuple[str, str]] = {
            f"{self.task_name(output,protocol)}": (output, protocol)
            for protocol in protocol_arms
            for output in self.output_names
        }

        self.tasks = list(self.full_tasks_map.keys())
        self.task_idx_to_output_idx: dict[int, int] = {
            self.tasks.index(task): self.output_names.index(output)
            for task, (output, _) in self.full_tasks_map.items()
        }
        self.task_idx_to_protocol: dict[int, str] = {
            self.tasks.index(task): protocol
            for task, (_, protocol) in self.full_tasks_map.items()
        }
        self.task_idx_to_protocol_idx: dict[int, int] = {
            self.tasks.index(task): self.protocol_arms.index(protocol)
            for task, (_, protocol) in self.full_tasks_map.items()
        }

    def task_name(self, output: str, protocol: str) -> str:
        return "_".join([output, protocol])

    def _validate_protocols(self, new_protocol_arms: list[str]) -> bool:
        return all(protocol in self.protocol_arms for protocol in new_protocol_arms)

    def _validate_ouputs(self, new_outputs: list[str]) -> bool:
        return all(output in self.output_names for output in new_outputs)

    def validate_tasks(self, new_protocol_arms: list[str], new_outputs: list[str]):
        """Validate a list of protocols and output names with respect to an existing task map."""
        if not self._validate_protocols(new_protocol_arms):
            raise ValueError(
                f"Incompatible protocol arms with supplied task map.\n{new_protocol_arms}\n{self.protocol_arms}"
            )
        if not self._validate_ouputs(new_outputs):
            raise ValueError(
                f"Incompatible output names with supplied task map.\n{new_outputs}\n{self.output_names}"
            )


def extend_schema(
    schema: pa.DataFrameSchema, column_list: list[str], type: str
) -> pa.DataFrameSchema:
    """Add user-specified columns to the training data schema."""
    if not column_list:
        return schema
    else:
        return schema.add_columns(
            {col: pa.Column(type, default=pd.NA) for col in column_list}
        )
