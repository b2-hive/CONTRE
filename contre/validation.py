import json
import b2luigi
import root_pandas
import basf2_mva
from contre.training import SplitSample, Training
from contre.weights import get_weights


@b2luigi.inherits(Training)
class ValidationExpert(b2luigi.Task):
    """Apply BDT to test samples and save result as `expert.root`.

    Parameters: See Training
    Output: expert.root
    """
    queue = "sx"

    def requires(self):
        yield self.clone_parent()

        # for test samples also require all split samples
        yield Training.requires(self)

    def output(self):
        yield self.add_to_output('expert.root')

    def run(self):
        bdt = self.get_input_file_names('bdt.xml')
        expert = self.get_output_file_name('expert.root')

        test_samples = self.get_input_file_names('test.root')

        basf2_mva.expert(
            basf2_mva.vector(*bdt),
            basf2_mva.vector(*test_samples),
            'ntuple', expert)


@b2luigi.inherits(ValidationExpert)
class ValidationReweighting(b2luigi.Task):
    """Calculate weights from the classifier output of the validation training.

    Parameters: see ValidationTraining
        normalize_to (float): Scale weights to match the ratio data / mc used
            for training.
    """
    normalize_to = b2luigi.FloatParameter()

    def requires(self):
        yield self.clone_parent()

    def output(self):
        yield self.add_to_output('weights.root')

    def run(self):
        expert = root_pandas.read_root(
            self.get_input_file_names('expert.root'))
        weights = get_weights(
            expert=expert,
            normalize_to=self.normalize_to)
        root_pandas.to_root(
            weights,
            self.get_output_file_name('weights.root'),
            key='weights')


class DelegateValidation(b2luigi.Task):
    """Delegate a validation training.

    Requires the SplitSample and ValidaionTraining tasks.
    Use Parameters stored in a json file.

    Parameters:
        name (str): Used for sorting, summarized results can be found in
            `<result_folder>/name=<name>/validation_results.json`
        parameter_file (str): name of the json file with stored settings,
    """
    name = b2luigi.Parameter()
    parameter_file = b2luigi.Parameter(significant=False)

    def requires(self):
        with open(self.parameter_file) as parameter_file:
            parameters = json.load(parameter_file)
            validation_parameters = parameters.get("validation_parameters")
        for off_res_file in parameters.get('off_res_files'):
            yield self.clone(
                SplitSample,
                ntuple_file=off_res_file,
                **validation_parameters
            )
        yield self.clone(
            Training,
            ntuple_files=parameters.get("off_res_files"),
            training_variables=parameters.get("training_variables"),
            **validation_parameters
        )
        yield self.clone(
            ValidationExpert,
            ntuple_files=parameters.get("off_res_files"),
            training_variables=parameters.get("training_variables"),
            **validation_parameters
        )
        yield self.clone(
            ValidationReweighting,
            ntuple_files=parameters.get("off_res_files"),
            training_variables=parameters.get("training_variables"),
            **validation_parameters,
            **parameters.get("reweighting_parameters")
        )

    def output(self):
        yield self.add_to_output('validation_results.json')

    def run(self):
        train_samples = self.get_input_file_names('train.root')
        test_samples = self.get_input_file_names('test.root')

        bdt = self.get_input_file_names('bdt.xml')
        expert = self.get_input_file_names('expert.root')
        weights = self.get_input_file_names('weights.root')

        results = {
            "parameter_file": self.parameter_file,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "bdt": bdt,
            "expert": expert,
            "weights": weights
        }

        with open(self.get_output_file_name(
                'validation_results.json'), 'w') as file:
            json.dump(results, file)
