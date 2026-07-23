import pandera.pandas as pa
import torch
import pandas as pd

from vpop_calibration.utils import extend_schema
from vpop_calibration.config import device, default_dtype
from vpop_calibration.pynlme.indexing import (
    DataIndex,
    ObservationsDataSet,
    SurvivalOutputs,
)
from vpop_calibration.pynlme.schemas import ObsDataSchema, patientDataSchema


class ObsData:
    def __init__(self, data: pa.typing.DataFrame):
        """Load and process an observed data set

        Args:
            data (pa.typing.DataFrame): The observed data. Should contain at least the columns ["id", "output_name", "time", "value"].
        """
        # Initial validation
        self.input_df = ObsDataSchema.validate(data)
        self.patients: list[str] = self.input_df.id.drop_duplicates().to_list()

        # Create the patient data frame (id, protocol_arm and descriptors)
        patients_df_raw = self.input_df.drop(
            columns=["output_name", "time", "value", "task"]
        ).drop_duplicates()
        self.descriptors_known: list[str] = [
            p
            for p in patients_df_raw.columns.to_list()
            if p
            not in ["id", "protocol_arm", "event_time", "event_status", "hazard_name"]
        ]

        patients_schema = extend_schema(
            patientDataSchema, self.descriptors_known, "float"
        )
        self.patients_df = patients_schema.validate(patients_df_raw)

        if "event_time" in self.patients_df.columns:
            # Process survival data
            self.hazard_name = self.patients_df["hazard_name"].drop_duplicates()
            assert self.hazard_name.shape[0] == 1, (
                f"More than one hazard name provided: {self.hazard_name}"
            )
            self.hazard_name = self.hazard_name.item()

            # The convention is that the model outputs for survival should be
            # `log_<hazard_name>` and `cumulative_<hazard_name>`
            self.survival_outputs = SurvivalOutputs(
                log_hazard="log_" + self.hazard_name,
                cumulative_hazard="cumulative_" + self.hazard_name,
            )
            self.survival_output_names = list(self.survival_outputs._asdict().values())

            surv_df_list = []
            # build the survival outcome rows
            for _, patient_row in self.patients_df.iterrows():
                new_row = patient_row[
                    ["id", "protocol_arm", "event_time", "event_status", "hazard_name"]
                ].rename({"event_time": "time", "event_status": "value"})
                individual_survival_df = pd.DataFrame([new_row, new_row]).astype(
                    {"value": float}
                )

                individual_survival_df["output_name"] = self.survival_output_names
                surv_df_list.append(
                    individual_survival_df[
                        ["id", "time", "protocol_arm", "output_name", "value"]
                    ]
                )
            self.surv_df = pd.concat(surv_df_list)
        else:
            self.surv_df = None
            self.hazard_name = None
            self.survival_output_names = []
            self.survival_outputs = None

        # Append the pivotted survival data to the input data frame
        self.input_df_with_survival = ObsDataSchema.validate(
            pd.concat(
                [
                    self.input_df[
                        ["id", "time", "output_name", "protocol_arm", "value"]
                    ],
                    self.surv_df,
                ]
            )
        )

        self.full_obs = ObservationsDataSet(
            obs_index=DataIndex.from_dataframe(self.input_df_with_survival),
            obs_values=torch.as_tensor(
                self.input_df_with_survival["value"].to_list(),
                device=device,
                dtype=default_dtype,
            ),
            survival_outputs=self.survival_outputs,
        )
        self.global_timesteps = torch.tensor(
            self.full_obs.obs_index.time.ref_values, device=device
        )
        self.nb_global_timesteps = self.global_timesteps.shape[0]
        self.nb_total_observations = self.full_obs.obs_values.shape[0]
        self.all_output_names = self.full_obs.obs_index.output_name.ref_values
        self.continuous_output_names = [
            output
            for output in self.all_output_names
            if output not in self.survival_output_names
        ]

        # Count the number of observations per output for variance scaling
        self.nb_tot_observations_per_output = torch.zeros(
            len(self.all_output_names), device=device
        )
        self.nb_tot_observations_per_output.scatter_add_(
            0,
            self.full_obs.obs_index.output_name.index_values,
            torch.ones_like(
                self.full_obs.obs_index.output_name.index_values,
                device=device,
                dtype=default_dtype,
            ),
        )

        self.individual_observations: dict[str, ObservationsDataSet] = {}
        for p in self.patients:
            patient_data = self.input_df_with_survival.loc[
                self.input_df_with_survival["id"] == p
            ]
            index_values_p = DataIndex.from_dataframe(patient_data)
            obs_values_p = torch.as_tensor(
                patient_data["value"].to_list(), device=device, dtype=default_dtype
            )
            self.individual_observations.update(
                {
                    p: ObservationsDataSet(
                        obs_index=index_values_p,
                        obs_values=obs_values_p,
                        survival_outputs=self.survival_outputs,
                    )
                }
            )

    def init_pdk_values(self, pdk_names: list[str]) -> None:
        """Generate per-patient PDK tensors

        Once initialized they are stored in `self.patients_pdk[patient_id]` and `self.patients_pdk_full`.

        Args:
            pdk_names (list[str]): The name of the known parameters which are to be assembled as pdk. Must appear in the data set columns.
        """
        assert set(pdk_names) <= set(self.descriptors_known), (
            f"Unknown PDK: {set(pdk_names) - set(self.descriptors_known)}"
        )
        self.pdk_names = pdk_names
        self.nb_pdk = len(pdk_names)
        self.patients_pdk = {}
        for patient in self.patients:
            if self.nb_pdk > 0:
                row = self.patients_df.loc[
                    self.patients_df["id"] == patient
                ].drop_duplicates()
                self.patients_pdk.update(
                    {
                        patient: torch.as_tensor(
                            row[self.pdk_names].values,
                            device=device,
                            dtype=default_dtype,
                        )
                    }
                )
            else:
                self.patients_pdk.update({patient: torch.empty((1, 0), device=device)})
        # Store the full pdk tensor on the device
        self.patients_pdk_full = torch.cat(
            [self.patients_pdk[ind] for ind in self.patients]
        ).to(device)

    def remap_all_indexings(
        self,
        new_patient_ids: list | None = None,
        new_output_names: list | None = None,
        new_protocol_arms: list | None = None,
        new_tasks: list | None = None,
        new_times: list | None = None,
    ):
        args = (
            new_patient_ids,
            new_output_names,
            new_protocol_arms,
            new_tasks,
            new_times,
        )
        self.full_obs.obs_index = self.full_obs.obs_index.remap_observation_index(*args)
        for p in self.patients:
            self.individual_observations[p].obs_index = self.individual_observations[
                p
            ].obs_index.remap_observation_index(*args)
