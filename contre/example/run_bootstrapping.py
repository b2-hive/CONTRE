import yaml
import b2luigi
from contre.bootstrapping import DelegateBootstrapping

parameter_file = 'example_parameters.yaml'
with open(parameter_file) as f:
    parameters = yaml.load(f)

b2luigi.set_setting(
    "result_path",
    parameters.get("result_path"),
)

b2luigi.process(
    DelegateBootstrapping(
        name=parameters.get("name"),
        parameter_file=parameter_file)
)
