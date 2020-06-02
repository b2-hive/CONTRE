import json
import yaml
import root_pandas
import basf2_mva
import b2luigi
from contre.weights import get_weights
from contre.training import Training, SplitSample
from contre.validation import ValidationReweighting, DelegateValidation


@b2luigi.requires(Training)
class Expert(b2luigi.Task):
    """Apply bdt to on resonance Data and save the result as `expert.root`.

    Parameters:
        off_res_files, tree_name, training_variables, training_parameters:
            see Training,
        on_res_files (list): List of str with on resonance files.
    """
    on_res_files = b2luigi.ListParameter(hashed=True)
    queue = "sx"

    def output(self):
        yield self.add_to_output("expert.root")

    def run(self):
        bdt = self.get_input_file_names('bdt.xml')
        expert = self.get_output_file_name('expert.root')

        basf2_mva.expert(
            basf2_mva.vector(*bdt),
            basf2_mva.vector(*self.on_res_files),
            self.tree_name, expert)


@b2luigi.inherits(Expert)
class Reweighting(b2luigi.Task):
    """Calculate weights from the classifier output of the validation training.

    Normalisaton of weights is taken from the validaton training.

    Parameters:
        off_res_files, on_res_files, training_variables, training_parameters:
            see Training,
    """

    def requires(self):
        yield self.clone_parent()
        yield self.clone(
            ValidationReweighting,
            off_res_files=self.off_res_files,
            training_variables=self.training_variables,
            **self.training_parameters
        )

    def output(self):
        yield self.add_to_output('weights.root')

    def run(self):
        # calculate the normalisation from the valadiaton reweighting
        validation_weights = root_pandas.read_root(
            self.get_input_file_names("validation_weights.root"))

        len_data = len(
            validation_weights[validation_weights["EventType"] == 1])
        len_mc = len(validation_weights) - len_data

        expert = root_pandas.read_root(
            self.get_input_file_names('expert.root'))
        weights = get_weights(
            expert_df=expert,
            normalize_to=len_data/len_mc)
        root_pandas.to_root(
            weights,
            self.get_output_file_name('weights.root'),
            key=self.tree_name)


class DelegateReweighting(b2luigi.Task):
    """Delegation Task for a Training and application to on resonance Files.

    Requires the Expert and Training task.
    Use Parameters stored in a json file.

    Parameters:
        name (str): Used for sorting, summarized results can be found in
            `<result_folder>/name=<name>/valiation_results.json`
        parameter_file (str): path to the json file with stored settings
    """

    name = b2luigi.Parameter()
    parameter_file = b2luigi.Parameter(significant=False)

    def requires(self):
        with open(self.parameter_file) as parameter_file:
            parameters = yaml.load(parameter_file)
        off_res_files = parameters.get("off_res_files")
        on_res_files = parameters.get("on_res_files")
        training_parameters = parameters.get("training_parameters")

        # require SplitSample Training and ValidationReweighting
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
            training_variables=parameters["training_variables"],
            tree_name=parameters["tree_name"],
            training_parameters=training_parameters
        )
        yield self.clone(
            ValidationReweighting,
            off_res_files=parameters.get("off_res_files"),
            tree_name=parameters["tree_name"],
            training_variables=parameters["training_variables"],
            training_parameters=training_parameters
        )

        if len(on_res_files) != 0:
            yield self.clone(
                Reweighting,
                off_res_files=off_res_files,
                on_res_files=on_res_files,
                tree_name=parameters["tree_name"],
                training_variables=parameters["training_variables"],
                training_parameters=training_parameters,
            )
        else:
            print(
                "No on-resonance files are given."
                "Only test samples will be reweighted.")

    def output(self):
        yield self.add_to_output('validation_results.json')
        yield self.add_to_output('results.json')

    def run(self):
        with open(self.parameter_file) as parameter_file:
            parameters = yaml.load(parameter_file)
        on_res_files = parameters.get("on_res_files")

        # write out file with the reweighted test samples
        DelegateValidation.run(self)
        bdt = self.get_input_file_names('bdt.xml')

        results = {
            "bdt": bdt,
        }
        if len(on_res_files) != 0:
            weights = self.get_input_file_names('weights.root')
            results['weights'] = weights

        with open(self.get_output_file_name(
                'results.json'), 'w') as file:
            json.dump(results, file)
