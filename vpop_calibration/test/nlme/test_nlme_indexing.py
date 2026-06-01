import torch
import pandas as pd

from vpop_calibration.pynlme.indexing import (
    IndexedValues,
    ObservationIndex,
    IndexedObservations,
)


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

    patient_indices = IndexedValues(
        index_values=torch.tensor([0, 0, 1, 1]),
        ref_values=patient_id,
        raw_values=pd.Series(["p1", "p1", "p2", "p2"]),
    )
    outputs_indices = IndexedValues(
        index_values=torch.tensor([0, 1, 0, 1]),
        ref_values=outputs,
        raw_values=pd.Series(["output_1", "output_2", "output_1", "output_2"]),
    )
    time_indices = IndexedValues(
        index_values=torch.tensor([0, 1, 2, 3]),
        ref_values=time,
        raw_values=pd.Series([0, 1, 2, 3]),
    )
    protocol_indices = IndexedValues(
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
    task_indices = IndexedValues(
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

    obs_index = ObservationIndex(
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

    patient_indices = IndexedValues(
        index_values=torch.tensor([0, 0, 1, 1]),
        ref_values=patient_id,
        raw_values=pd.Series(["p1", "p1"]),
    )
    outputs_indices = IndexedValues(
        index_values=torch.tensor([0, 1, 0, 1]),
        ref_values=outputs,
        raw_values=pd.Series(["output_1", "output_2", "output_1", "output_2"]),
    )
    time_indices = IndexedValues(
        index_values=torch.tensor([0, 1, 2, 3]),
        ref_values=time,
        raw_values=pd.Series([0, 1, 2, 3]),
    )
    protocol_indices = IndexedValues(
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
    task_indices = IndexedValues(
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

    obs_index = ObservationIndex(
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

    patient_indices = IndexedValues(
        index_values=torch.tensor([0, 0, 1, 1]),
        ref_values=patient_id,
        raw_values=pd.Series(["p1", "p1", "p2", "p2"]),
    )
    outputs_indices = IndexedValues(
        index_values=torch.tensor([0, 1, 0, 1]),
        ref_values=outputs,
        raw_values=pd.Series(["output_1", "output_2", "output_1", "output_2"]),
    )
    time_indices = IndexedValues(
        index_values=torch.tensor([0, 1, 2, 3]),
        ref_values=time,
        raw_values=pd.Series([0, 1, 2, 3]),
    )
    protocol_indices = IndexedValues(
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
    task_indices = IndexedValues(
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

    obs_index = ObservationIndex(
        id=patient_indices,
        output_name=outputs_indices,
        protocol_arm=protocol_indices,
        time=time_indices,
        task=task_indices,
    )
    obs_values = torch.tensor([0, 1, 2, 3])
    pred_values = torch.tensor([[4, 5, 6, 7]])
    indexed_obs = IndexedObservations(obs_index=obs_index, obs_values=obs_values)

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
    pd.testing.assert_frame_equal(df, expected_df)


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

    obs_index = ObservationIndex.from_dataframe(df_in)
    value = torch.as_tensor(df_in.value.values)
    indexed_value = IndexedObservations(obs_index=obs_index, obs_values=value)
    df_out = indexed_value.to_pandas()

    pd.testing.assert_frame_equal(df_in.drop(columns=["task"]), df_out)
