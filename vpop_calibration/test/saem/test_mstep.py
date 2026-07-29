from vpop_calibration.saem.m_step import MStepState

import torch


def test_mstep():
    nb_pdu = 3
    indiv_design_matrix = torch.eye(nb_pdu).double()
    nb_chains = 1
    nb_patients = 2
    design_matrix = torch.stack((indiv_design_matrix, indiv_design_matrix), dim=0)

    gaussian_params = torch.zeros((nb_chains, nb_patients, nb_pdu)).double()

    state = MStepState.from_init_gaussian_params(
        design_matrix=design_matrix,
        nb_chains=nb_chains,
        nb_patients=nb_patients,
        nb_pdu=nb_pdu,
        init_gaussian_params=gaussian_params,
    )

    new_gaussian_params = torch.ones((nb_chains, nb_patients, nb_pdu)).double()
    _proposal = state.update(new_gaussian_params=new_gaussian_params, learning_rate=0.1)


def test_state_dict():
    nb_pdu = 3
    indiv_design_matrix = torch.eye(nb_pdu).double()
    nb_chains = 1
    nb_patients = 2
    design_matrix = torch.stack((indiv_design_matrix, indiv_design_matrix), dim=0)

    gaussian_params = torch.zeros((nb_chains, nb_patients, nb_pdu)).double()

    state = MStepState.from_init_gaussian_params(
        design_matrix=design_matrix,
        nb_chains=nb_chains,
        nb_patients=nb_patients,
        nb_pdu=nb_pdu,
        init_gaussian_params=gaussian_params,
    )
    state_dict = state.get_state_dict()
    new_mstep_state = MStepState.from_state_dict(state_dict=state_dict)

    assert state == new_mstep_state
