import torch

from vpop_calibration.nlme_model.indexing import IndexedValues, ObservationIndex


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
    ]

    patient_indices = IndexedValues(torch.tensor([0, 0, 1, 1]), patient_id)
    outputs_indices = IndexedValues(torch.tensor([0, 1, 0, 1]), outputs)
    time_indices = IndexedValues(torch.tensor([0, 1, 2, 3]), time)
    protocol_indices = IndexedValues(torch.tensor([0, 1, 0, 2]), protocols)
    task_indices = IndexedValues(torch.tensor([0, 1, 2, 3]), tasks)

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
        new_obs_index.task.index_values, torch.tensor([2, 3, 0, 1])
    )
