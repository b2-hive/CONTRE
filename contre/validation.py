import json
import b2luigi
import root_pandas
import basf2_mva
from contre.training import SplitSample, Training
from contre.weights import get_weights


@b2luigi.inherits(Training)
class ValidationExpert(b2luigi.Task):
    """Apply BDT to test samples and save result as `validaion_expert.root`.

    Parameters:
        off_res_files, tree_name, training_variables, training_parameters:
            see Training.
    """
    queue = "sx"

    def requires(self):
        yield self.clone_parent()

        # for test samples also require all split samples
        yield Training.requires(self)

    def output(self):
        yield self.add_to_output('validation_expert.root')

    def run(self):
        bdt = self.get_input_file_names('bdt.xml')
        expert = self.get_output_file_name('validation_expert.root')

        test_samples = self.get_input_file_names('test.root')

        basf2_mva.expert(
            basf2_mva.vector(*bdt),
            basf2_mva.vector(*test_samples),
            self.tree_name, expert)


@b2luigi.inherits(ValidationExpert)
class ValidationReweighting(b2luigi.Task):
    """Calculate weights from the classifier output of the ValidationTraining
    task. See also: Reweighting.

    Parameters:
        off_res_files, tree_name, training_variables, training_parameters:
            see Training.
    """

    def requires(self):
        yield self.clone_parent()

    def output(self):
        yield self.add_to_output('validation_weights.root')

    def run(self):
        expert = root_pandas.read_root(
            self.get_input_file_names('validation_expert.root'))

        # normalize to len_data / len_mc (off-res.)
        key_EventType = expert.keys()[1]
        len_data = len(
            expert[expert[key_EventType] == 1])
        len_mc = len(expert) - len_data

        weights = get_weights(
            expert_df=expert,
            normalize_to=len_data/len_mc)

        root_pandas.to_root(
            weights,
            self.get_output_file_name('validation_weights.root'),
            key='weights')


class DelegateValidation(b2luigi.Task):
    """Delegate reweighting of an off-resonance test sample.

    Starts the SplitSample, Training and ValidationReweighting tasks.
    Uses parameters from the parameter file.

    Parameters:
        name (str): Used for sorting, summarized results can be found in
            `<result_folder>/name=<name>/validation_results.json`,
        parameter_file (str): name of the parameter file.
    """
    name = b2luigi.Parameter()
    parameter_file = b2luigi.Parameter(significant=False)

    def requires(self):
        with open(self.parameter_file) as parameter_file:
            parameters = json.load(parameter_file)
            training_parameters = parameters.get("training_parameters")
        for off_res_file in parameters.get('off_res_files'):
            yield self.clone(
                SplitSample,
                ntuple_file=off_res_file,
                tree_name=parameters["tree_name"],
                train_size=training_parameters.get("train_size"),
                test_size=training_parameters.get("test_size")
            )
        yield self.clone(
            Training,
            off_res_files=parameters.get("off_res_files"),
            tree_name=parameters["tree_name"],
            training_parameters=training_parameters
        )
        yield self.clone(
            ValidationReweighting,
            off_res_files=parameters.get("off_res_files"),
            tree_name=parameters["tree_name"],
            training_parameters=training_parameters,
            )

    def output(self):
        yield self.add_to_output('validation_results.json')

    def run(self):
        train_samples = self.get_input_file_names('train.root')
        test_samples = self.get_input_file_names('test.root')

        bdt = self.get_input_file_names('bdt.xml')
        validation_weights = self.get_input_file_names(
            'validation_weights.root')

        results = {
            "parameter_file": self.parameter_file,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "bdt": bdt,
            "validation_weights": validation_weights
        }

        with open(self.get_output_file_name(
                'validation_results.json'), 'w') as file:
            json.dump(results, file)
