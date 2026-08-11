import pandas as pd
import numpy as np
import time
import torch
from deepdiff import DeepDiff

from vpop_calibration import *

## Loading dataset
df = pd.read_csv("Mavoglurant_Dataset.csv")

##Loading computational model
## libRoadRunner
sbml_model = StructuralSbml(
    model_path="CM_Mavoglurant.xml",
    inputs=["KbBR", "CLint", "KbMU", "KbAD", "KbBO", "KbRB", "Dose", "WT"],
    outputs=["logC15"],
)

sbml_model.rr.setIntegrator("cvode")
integ = sbml_model.rr.integrator
integ.setValue("stiff", True)
integ.setValue("relative_tolerance", 1e-6)
integ.setValue("absolute_tolerance", 1e-6)
integ.setValue("initial_time_step", 1e-6)
integ.setValue("maximum_time_step", 500.0)

## Simwork
simwork_model = SimworkModelBinding(
    path_to_model="cm.json",
    path_to_solving_options="sv.json",
    inputs=["KbBR", "CLint", "KbMU", "KbAD", "KbBO", "KbRB", "Dose", "WT"],
    outputs=["logC15"],
)

## Initial Estimates
prior = {
    "pdu": {
        "CLint": {"prior": np.exp(7.6), "prior_omega": 4},
        "KbBR": {"prior": np.exp(1.1), "prior_omega": 0.5},
        "KbMU": {"prior": np.exp(0.3), "prior_omega": 0.5},
        "KbBO": {"prior": np.exp(0.03), "prior_omega": 0.5},
        "KbAD": {"prior": np.exp(2), "prior_omega": 0.5},
        "KbRB": {"prior": np.exp(0.3), "prior_omega": 0.5},
    },
    "pdk": {"WT", "Dose"},
    "error_model": {
        "logC15": {"error_type": "additive", "sigma": 0.5},
    },
}

## Config
config = Config(
    saem=SaemConfigDict(
        nb_iter_burnin=10,
        nb_iter_learning=10,
        nb_iter_smoothing=10,
    ),
    nlme=NlmeConfigDict(nb_chains=1),
)


# Sélectionne une quantité de patients
def build_population_obs(df, n_patients=1, dose_col="Dose", ids=None):
    if ids is None:
        ids = sorted(df["ID"].unique())[:n_patients]
    d = df.loc[df["ID"].isin(ids) & (df["EVID"] == 0)]
    obs = d.rename(
        columns={"ID": "id", "TIME": "time", "DV": "value", "DOSE": dose_col}
    )[["id", "time", "value", dose_col, "WT"]].astype({"value": "float"})
    obs["time"] = obs["time"] * 3600.0
    obs["value"] = np.log(obs["value"])
    obs["output_name"] = "logC15"
    obs["protocol_arm"] = "identity"
    return obs, list(ids)


# Runtime benchmark per iteration of a full run of SAEM
def benchmark_runtime(structural_model, obs_df):
    nlme_model = NlmeModel(
        df=obs_df,
        prior_params=prior,
        structural_model=structural_model,
        config=config,
    )

    opt = nlme_model.optimizer

    t0 = time.perf_counter()
    if opt.scheduler.iteration == 0:
        opt.init_state()
    t_init = time.perf_counter() - t0

    nb_b = opt.config.nb_iter_burnin
    nb_l = opt.config.nb_iter_learning

    def phase_of(it):
        if it < nb_b:
            return "burnin"
        if it < nb_b + nb_l:
            return "learning"
        return "smoothing"

    records = []
    prev = time.perf_counter()
    for summary in opt.optimization_stream():
        now = time.perf_counter()
        records.append(
            {
                "iteration": summary.iteration,
                "phase": phase_of(summary.iteration),
                "time_s": now - prev,
            }
        )
        prev = now
    per_iter = pd.DataFrame(records)

    per_phase = (
        per_iter.groupby("phase")["time_s"]
        .agg(n="count", total_s="sum", mean_s="mean", median_s="median", std_s="std")
        .reindex(["burnin", "learning", "smoothing"])
        .dropna(how="all")
    )
    summary = {
        "init_s": t_init,
        "iteration_total_s": per_iter["time_s"].sum(),
        "total_s": t_init + per_iter["time_s"].sum(),
        "first_iter_s": per_iter["time_s"].iloc[0],
        "per_phase": per_phase,
    }
    return summary, per_iter, nlme_model


def _median_time(fn, repeats=5, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


torch.set_num_threads(1)

# Runtime pour 1 seul patient
n_patients = 1
_, ids = build_population_obs(df, n_patients=n_patients)

obs_sbml, _ = build_population_obs(df, ids=ids, dose_col="Dose")
obs_sim, _ = build_population_obs(df, ids=ids, dose_col="Dose")
print(f"n_patients={len(ids)}  n_obs={obs_sbml.shape[0]}")

print("\n### SBML : benchmark_runtime")
rt_sbml, iters_sbml, nlme_sbml = benchmark_runtime(sbml_model, obs_sbml)
print(rt_sbml["per_phase"])
print("total:", round(rt_sbml["total_s"], 3), "s")

print("\n### Simwork : benchmark_runtime")
rt_sim, iters_sim, nlme_sim = benchmark_runtime(
    StructuralSimwork(model=simwork_model), obs_sim
)
print(rt_sim["per_phase"])
print("total:", round(rt_sim["total_s"], 3), "s")


# Coût d'un E-step sur un NlmeModel
def time_predict_all_patients(structural_model, obs_df, repeats=5, warmup=1):
    nlme = NlmeModel(
        df=obs_df, prior_params=prior, structural_model=structural_model, config=config
    )
    opt = nlme.optimizer
    opt.init_state()
    model = nlme.statistical_model

    etas = model.sample_etas(model.nb_chains)
    gaussian_params = model.convert_etas_to_gaussian_all_patients(etas)
    physical_params = model.convert_gaussian_to_physical(gaussian_params, model.log_mi)
    thetas = model.convert_physical_to_thetas_all_patients(physical_params)
    inputs = model.convert_thetas_to_model_parameters_all_patients(thetas)

    def one():
        # Benchmark only the true function of interest
        return model.predict_all_patients(inputs)

    return _median_time(one, repeats, warmup)


n_list = [1, 5, 20, 50, 100]
n_list = [n for n in n_list if n <= df["ID"].nunique()]

rows = []
for n in n_list:
    _, ids = build_population_obs(df, n_patients=n)
    obs = build_population_obs(df, ids=ids, dose_col="Dose")[0]
    n_obs = obs.shape[0]

    t_sbml = time_predict_all_patients(sbml_model, obs)
    t_sim = time_predict_all_patients(StructuralSimwork(model=simwork_model), obs)

    rows.append(
        {
            "n_patients": n,
            "n_obs": n_obs,
            "sbml_ms": t_sbml * 1e3,
            "simwork_ms": t_sim * 1e3,
            "ratio": t_sim / t_sbml,
        }
    )
    print(
        f"N={n:4d}  n_obs={n_obs:4d}  |  SBML {t_sbml * 1e3:7.2f} ms  |  "
        f"Simwork {t_sim * 1e3:8.2f} ms  |  x{t_sim / t_sbml:.0f}"
    )

scaling = pd.DataFrame(rows)


def fit(col):
    slope, intercept = np.polyfit(scaling["n_patients"], scaling[col], 1)
    return slope, intercept


for eng in ["sbml_ms", "simwork_ms"]:
    s, i = fit(eng)
    print(f"{eng:11s}: {i:8.2f} ms fixe + {s:7.3f} ms/patient")


# Prédictions déterministes à etas figés, structurées {patient_id: [valeurs]}.
def _predictions_by_patient(nlme_model) -> dict:
    model = nlme_model.statistical_model
    etas = model.sample_etas(model.nb_chains)
    pred = model.log_posterior_etas_all_patients(etas).predictions
    pred = pred.detach().cpu()

    out = {}
    id_index = model.data.full_obs.obs_index.id
    for i, patient_id in enumerate(id_index.ref_values):
        rows = id_index.index_values == i
        out[str(patient_id)] = pred[:, rows].numpy().tolist()
    return out


# Comparaison des prédictions déterministes du modèle structurel
def test_structural_predictions_match_across_engines(math_epsilon=1e-5):
    _, ids = build_population_obs(df, n_patients=1)
    obs = build_population_obs(df, ids=ids, dose_col="Dose")[0]

    nlme_sbml = NlmeModel(
        df=obs, prior_params=prior, structural_model=sbml_model, config=config
    )
    nlme_sim = NlmeModel(
        df=obs,
        prior_params=prior,
        structural_model=StructuralSimwork(model=simwork_model),
        config=config,
    )

    actual = _predictions_by_patient(nlme_sim)
    expected = _predictions_by_patient(nlme_sbml)

    assert actual.keys() == expected.keys(), (
        f"patients différents : {actual.keys()} vs {expected.keys()}"
    )

    diff = DeepDiff(
        actual,
        expected,
        ignore_type_in_groups=[(tuple, list), (float, np.float64)],
        math_epsilon=math_epsilon,
    )
    assert diff == {}, f"prédictions divergentes au-delà de {math_epsilon}:\n{diff}"


if __name__ == "__main__":
    test_structural_predictions_match_across_engines()
