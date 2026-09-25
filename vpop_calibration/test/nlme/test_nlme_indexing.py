import torch
import pandas as pd
import pytest

from vpop_calibration.pynlme.indexing import (
    TensorIndexing,
    DataIndex,
    ObservationsDataSet,
    SurvivalOutputs,
    remap_single_index,
)
from vpop_calibration.pynlme.schemas import ObsDataSchema


def test_observation_indexing():
    patient_id: list = ["p1", "p2"]
    protocols: list = ["arm-1", "arm-2", "arm-3"]
    outputs: list = ["output_1", "output_2"]
    time: list = [0, 1, 2, 3]
    tasks: list = [
        "output_1_arm-1",
        "output_2_arm-1",
        "output_1_arm-2",
        "output_2_arm-2",
        "output_1_arm-3",
        "output_2_arm-3",
    ]

    patient_indices = TensorIndexing(
        index_values=torch.tensor([0, 0, 1, 1]),
        ref_values=patient_id,
        raw_values=pd.Series(["p1", "p1", "p2", "p2"]),
    )
    outputs_indices = TensorIndexing(
        index_values=torch.tensor([0, 1, 0, 1]),
        ref_values=outputs,
        raw_values=pd.Series(["output_1", "output_2", "output_1", "output_2"]),
    )
    time_indices = TensorIndexing(
        index_values=torch.tensor([0, 1, 2, 3]),
        ref_values=time,
        raw_values=pd.Series([0, 1, 2, 3]),
    )
    protocol_indices = TensorIndexing(
        index_values=torch.tensor([0, 1, 0, 2]),
        ref_values=protocols,
        raw_values=pd.Series(
            [
                "arm-1",
                "arm-2",
                "arm-1",
                "arm-3",
            ]
        ),
    )
    task_indices = TensorIndexing(
        index_values=torch.tensor([0, 3, 0, 5]),
        ref_values=tasks,
        raw_values=pd.Series(
            [
                "output_1_arm-1",
                "output_2_arm-2",
                "output_1_arm-1",
                "output_2_arm-3",
            ]
        ),
    )

    obs_index = DataIndex(
        id=patient_indices,
        output_name=outputs_indices,
        protocol_arm=protocol_indices,
        time=time_indices,
        task=task_indices,
    )
    torch.testing.assert_close(obs_index.id.index_values, torch.tensor([0, 0, 1, 1]))
    torch.testing.assert_close(
        obs_index.output_name.index_values, torch.tensor([0, 1, 0, 1])
    )
    torch.testing.assert_close(obs_index.time.index_values, torch.tensor([0, 1, 2, 3]))
    torch.testing.assert_close(
        obs_index.protocol_arm.index_values, torch.tensor([0, 1, 0, 2])
    )
    torch.testing.assert_close(obs_index.task.index_values, torch.tensor([0, 3, 0, 5]))


def test_remapping():
    patient_id: list = ["p1", "p2"]
    protocols: list = ["arm-1", "arm-2", "arm-3"]
    outputs: list = ["output_1", "output_2"]
    time: list = [0, 1, 2, 3]
    tasks: list = [
        "output_1_arm-1",
        "output_2_arm-1",
        "output_1_arm-2",
        "output_2_arm-2",
        "output_1_arm-3",
        "output_2_arm-3",
    ]

    patient_indices = TensorIndexing(
        index_values=torch.tensor([0, 0, 1, 1]),
        ref_values=patient_id,
        raw_values=pd.Series(["p1", "p1"]),
    )
    outputs_indices = TensorIndexing(
        index_values=torch.tensor([0, 1, 0, 1]),
        ref_values=outputs,
        raw_values=pd.Series(["output_1", "output_2", "output_1", "output_2"]),
    )
    time_indices = TensorIndexing(
        index_values=torch.tensor([0, 1, 2, 3]),
        ref_values=time,
        raw_values=pd.Series([0, 1, 2, 3]),
    )
    protocol_indices = TensorIndexing(
        index_values=torch.tensor([0, 1, 0, 2]),
        ref_values=protocols,
        raw_values=pd.Series(
            [
                "arm_1",
                "arm_2",
                "arm_1",
                "arm_3",
            ]
        ),
    )
    task_indices = TensorIndexing(
        index_values=torch.tensor([0, 3, 0, 5]),
        ref_values=tasks,
        raw_values=pd.Series(
            [
                "output_1_arm-1",
                "output_2_arm-2",
                "output_1_arm-1",
                "output_2_arm-3",
            ]
        ),
    )

    obs_index = DataIndex(
        id=patient_indices,
        output_name=outputs_indices,
        protocol_arm=protocol_indices,
        time=time_indices,
        task=task_indices,
    )

    new_patient_ids: list = ["p2", "p1"]
    new_protocols: list = ["arm-3", "arm-2", "arm-1"]
    new_outputs: list = outputs
    new_time: list = [3, 1, 0, 2]
    new_tasks: list = [
        "output_1_arm-2",
        "output_2_arm-2",
        "output_1_arm-1",
        "output_2_arm-1",
        "output_1_arm-3",
        "output_2_arm-3",
    ]

    new_obs_index = obs_index.remap_observation_index(
        new_patient_ids=new_patient_ids,
        new_output_names=new_outputs,
        new_protocol_arms=new_protocols,
        new_times=new_time,
        new_tasks=new_tasks,
    )

    torch.testing.assert_close(
        new_obs_index.id.index_values, torch.tensor([1, 1, 0, 0])
    )
    torch.testing.assert_close(
        new_obs_index.protocol_arm.index_values, torch.tensor([2, 1, 2, 0])
    )
    torch.testing.assert_close(
        new_obs_index.output_name.index_values, torch.tensor([0, 1, 0, 1])
    )
    torch.testing.assert_close(
        new_obs_index.time.index_values, torch.tensor([2, 1, 3, 0])
    )
    torch.testing.assert_close(
        new_obs_index.task.index_values, torch.tensor([2, 1, 2, 5])
    )


def test_to_pandas():
    patient_id: list = ["p1", "p2"]
    protocols: list = ["arm-1", "arm-2", "arm-3"]
    outputs: list = ["output_1", "output_2"]
    time: list = [0, 1, 2, 3]
    tasks: list = [
        "output_1_arm-1",
        "output_2_arm-1",
        "output_1_arm-2",
        "output_2_arm-2",
        "output_1_arm-3",
        "output_2_arm-3",
    ]

    patient_indices = TensorIndexing(
        index_values=torch.tensor([0, 0, 1, 1]),
        ref_values=patient_id,
        raw_values=pd.Series(["p1", "p1", "p2", "p2"]),
    )
    outputs_indices = TensorIndexing(
        index_values=torch.tensor([0, 1, 0, 1]),
        ref_values=outputs,
        raw_values=pd.Series(["output_1", "output_2", "output_1", "output_2"]),
    )
    time_indices = TensorIndexing(
        index_values=torch.tensor([0, 1, 2, 3]),
        ref_values=time,
        raw_values=pd.Series([0, 1, 2, 3]),
    )
    protocol_indices = TensorIndexing(
        index_values=torch.tensor([0, 1, 0, 2]),
        ref_values=protocols,
        raw_values=pd.Series(
            [
                "arm-1",
                "arm-2",
                "arm-1",
                "arm-3",
            ]
        ),
    )
    task_indices = TensorIndexing(
        index_values=torch.tensor([0, 1, 2, 3]),
        ref_values=tasks,
        raw_values=pd.Series(
            [
                "output_1_arm-1",
                "output_2_arm-2",
                "output_1_arm-1",
                "output_2_arm-3",
            ]
        ),
    )

    obs_index = DataIndex(
        id=patient_indices,
        output_name=outputs_indices,
        protocol_arm=protocol_indices,
        time=time_indices,
        task=task_indices,
    )
    obs_values = torch.tensor([0, 1, 2, 3])
    pred_values = torch.tensor([[4, 5, 6, 7]])
    indexed_obs = ObservationsDataSet(obs_index=obs_index, obs_values=obs_values)

    df = indexed_obs.to_pandas(prediction=pred_values)
    expected_df = pd.DataFrame(
        {
            "id": ["p1", "p1", "p2", "p2"],
            "output_name": ["output_1", "output_2", "output_1", "output_2"],
            "protocol_arm": ["arm-1", "arm-2", "arm-1", "arm-3"],
            "time": [0, 1, 2, 3],
            "value": [0, 1, 2, 3],
            "predicted_value": [4, 5, 6, 7],
        }
    )
    pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)


def test_from_pandas():
    patient_id: list = ["p1", "p2"]
    protocols: list = ["arm-1", "arm-2", "arm-3"]
    outputs: list = ["output_1", "output_2"]
    time: list = [0, 1, 2, 3]

    df_in = (
        pd.DataFrame({"id": patient_id})
        .merge(pd.DataFrame({"output_name": outputs}), how="cross")
        .merge(pd.DataFrame({"protocol_arm": protocols}), how="cross")
        .merge(pd.DataFrame({"time": time}), how="cross")
    )
    df_in["value"] = range(df_in.shape[0])
    df_in["task"] = df_in.apply(
        lambda r: r["output_name"] + "_" + r["protocol_arm"], axis=1
    )
    df_in_val = ObsDataSchema.validate(df_in)
    obs_index = DataIndex.from_dataframe(df_in_val)
    value = torch.as_tensor(df_in_val.value.values)
    indexed_value = ObservationsDataSet(obs_index=obs_index, obs_values=value)
    df_out = indexed_value.to_pandas()

    pd.testing.assert_frame_equal(
        df_in_val.drop(columns=["task"]), df_out, check_dtype=False
    )


@pytest.fixture
def sparse_observations():
    df = pd.DataFrame(
        {
            "id": ["p2", "p0", "p3", "p2", "p1", "p0"],
            "output_name": ["a", "c", "c", "b", "a", "c"],
            "protocol_arm": ["arm-0", "arm-2", "arm-2", "arm-1", "arm-0", "arm-2"],
            "task": ["a_arm-0", "c_arm-2", "c_arm-2", "b_arm-1", "a_arm-0", "c_arm-2"],
            "time": [0, 2, 2, 1, 0, 3],
        },
        index=[20, 30, 40, 50, 60, 70],
    )
    return ObservationsDataSet(
        obs_index=DataIndex.from_dataframe(df),
        obs_values=torch.tensor([20.0, 30.0, 40.0, 50.0, 60.0, 70.0]),
        survival_outputs=SurvivalOutputs("log_hazard", "cumulative_hazard"),
    )


def test_select_patients_preserves_sparse_rows_and_model_indices(sparse_observations):
    selected = sparse_observations.select_patients(torch.tensor([3, 0]))
    expected_positions = [1, 2, 5]

    assert selected.obs_index.id.ref_values == ["p3", "p0"]
    torch.testing.assert_close(
        selected.obs_index.id.index_values,
        torch.tensor(
            [1, 0, 1], device=sparse_observations.obs_index.id.index_values.device
        ),
    )
    torch.testing.assert_close(selected.obs_values, torch.tensor([30.0, 40.0, 70.0]))
    for field in DataIndex._fields:
        original_index = getattr(sparse_observations.obs_index, field)
        selected_index = getattr(selected.obs_index, field)
        pd.testing.assert_series_equal(
            selected_index.raw_values,
            original_index.raw_values.iloc[expected_positions],
        )
        if field != "id":
            assert selected_index.ref_values == original_index.ref_values
            torch.testing.assert_close(
                selected_index.index_values,
                original_index.index_values[expected_positions],
            )
    assert selected.survival_outputs == sparse_observations.survival_outputs


def test_select_patients_does_not_mutate_source(sparse_observations):
    original = sparse_observations.model_copy(deep=True)
    selected = sparse_observations.select_patients(torch.tensor([3, 0]))
    selected.obs_values.fill_(-1)
    for index in selected.obs_index:
        index.index_values.fill_(0)
        index.ref_values.clear()
        index.raw_values.iloc[:] = index.raw_values.iloc[0]

    torch.testing.assert_close(sparse_observations.obs_values, original.obs_values)
    for original_index, actual_index in zip(
        original.obs_index, sparse_observations.obs_index
    ):
        torch.testing.assert_close(
            actual_index.index_values, original_index.index_values
        )
        assert actual_index.ref_values == original_index.ref_values
        pd.testing.assert_series_equal(
            actual_index.raw_values, original_index.raw_values
        )


@pytest.mark.parametrize(
    "patient_indices, message",
    [
        (torch.tensor([], dtype=torch.long), "nonempty one-dimensional"),
        (torch.tensor(0), "nonempty one-dimensional"),
        (torch.tensor([[0, 1]]), "nonempty one-dimensional"),
        (torch.tensor([0.0, 1.0]), "integer"),
        (torch.tensor([True, False]), "integer"),
        (torch.tensor([1, 1]), "unique"),
        (torch.tensor([-1, 0]), "out-of-range"),
        (torch.tensor([0, 4]), "out-of-range"),
    ],
)
def test_select_patients_rejects_invalid_indices(
    sparse_observations, patient_indices, message
):
    with pytest.raises(ValueError, match=message):
        sparse_observations.select_patients(patient_indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "observation_device, selection_device", [("cuda", "cpu"), ("cpu", "cuda")]
)
def test_select_patients_handles_selection_on_another_device(
    sparse_observations, observation_device, selection_device
):
    sparse_observations.obs_index = DataIndex(
        *[
            index._replace(index_values=index.index_values.to(observation_device))
            for index in sparse_observations.obs_index
        ]
    )
    sparse_observations.obs_values = sparse_observations.obs_values.to(
        observation_device
    )
    selected = sparse_observations.select_patients(
        torch.tensor([3, 0], device=selection_device)
    )
    for index in selected.obs_index:
        assert index.index_values.device.type == observation_device
    torch.testing.assert_close(
        selected.obs_values, torch.tensor([30.0, 40.0, 70.0], device=observation_device)
    )
    torch.testing.assert_close(
        selected.obs_index.id.index_values,
        torch.tensor([1, 0, 1], device=observation_device),
    )
