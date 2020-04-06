import b2luigi
from contre.validation import DelegateValidation

parameters_file = 'example.json'

b2luigi.set_setting(
    "result_path",
    parameters_file.get("result_path"),
)

b2luigi.process(
    DelegateValidation,
    name=parameters_file.get("name"),
    parameters_file=parameters_file
)
