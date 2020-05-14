import json
import b2luigi
from contre.validation import DelegateValidation
from contre.reweighting import DelegateReweighting

parameter_file = 'example_parameters.json'
with open(parameter_file) as file:
    parameters = json.load(file)

b2luigi.set_setting(
    "result_path",
    parameters.get("result_path"),
)

b2luigi.process(
    DelegateReweighting(
        name=parameters.get("name"),
        parameter_file=parameter_file)
)
