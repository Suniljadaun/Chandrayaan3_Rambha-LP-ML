"""
rambhalp — physics-informed ML inversion of Chandrayaan-3 RAMBHA-LP Langmuir sweeps.

Public modules:
    config      load config.yaml
    io_pds      read the PDS4 raw/rawA/rawB archive
    preprocess  segment + bin sweeps into clean I-V curves
    classical   classical OML baseline fit + failure classification
    physics     OML forward model + physics-informed loss
    synthetic   labelled synthetic sweep generator
    model       the PyTorch inversion network
    train       training loop
    maven       MAVEN pretrain + transfer
    infer       apply the model to real sweeps
    validate    hop blind-test + uncertainty
    figures     all report figures
"""
__version__ = "1.0.0"
