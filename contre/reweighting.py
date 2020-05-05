import json
import root_pandas
import basf2_mva
import b2luigi
from contre.weights import get_weights
from contre.training import Training


@b2luigi.requires(Training)
class Expert(b2luigi.Task):
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
            'ntuple', expert)


@b2luigi.inherits(Expert)
class Reweighting(b2luigi.Task):
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
            parameters = json.load(parameter_file)
        off_res_files = parameters.get("off_res_files")
        on_res_files = parameters.get("on_res_files")
        training_variables = parameters.get("training_variables")

        yield self.clone(
            Training,
            off_res_files=off_res_files,
            training_variables=training_variables,
        )
        if len(on_res_files) != 0:
            yield self.clone(
                Reweighting,
                off_res_files=off_res_files,
                on_res_files=on_res_files,
                training_variables=training_variables,
                **parameters.get("reweighting_parameters")
            )
        else:
            print(
                "No on-resonance files are given."
                "Only training  and no reweighting can be executed.")

    def output(self):
        yield self.add_to_output('validation_results.json')

    def run(self):
        bdt = self.get_input_file_names('bdt.xml')

        results = {
            "bdt": bdt,
        }
        if len(self.on_res_files) != 0:
            weights = self.get_input_file_names('weights.root')
            results['weights'] = weights

        with open(self.get_output_file_name(
                'validation_results.json'), 'w') as file:
            json.dump(results, file)


class ExpertFromXml(b2luigi.Task):
    """Apply expert on On_res_files, without training beforehand.

    Parameters:
        bdt (str): Path to saved bdt weightfile
        on_res_files (list): List with ntuple file.
    """
    bdt = b2luigi.Parameter()
    on_res_files = b2luigi.ListParameter(hashed=True)

    def output(self):
        yield self.add_to_output("expert.root")

    def run(self):
        expert = self.get_output_file_name('expert.root')

        basf2_mva.expert(
            basf2_mva.vector(self.bdt),
            basf2_mva.vector(*self.on_res_files),
            'ntuple', expert)
