import json
import yaml
import b2luigi
import root_pandas
from sklearn.utils import resample
from contre.training import SplitSample, Training
from contre.reweighting import Expert, Reweighting
from contre.validation import ValidationExpert, ValidationReweighting


@b2luigi.requires(SplitSample)
class Resample(b2luigi.Task):
    """Resample the train sample and store it to a root file.

    Parameters:
        ntuple_file (str): Path to the file
        train_size (float): between 0 and 1, size of train sample
        test_size (float): size of test sample,
        random_seed (int): random seed to generate a resampled sample

    Output:
        train.root
    """
    random_seed = b2luigi.IntParameter()

    def output(self):
        yield self.add_to_output("train.root")

    def run(self):
        df = root_pandas.read_root(
            *self.get_input_file_names('train.root'),
            key=self.tree_name)

        # resample
        resampled_df = resample(df, random_state=self.random_seed)

        # store to root
        root_pandas.to_root(
            resampled_df,
            self.get_output_file_name('train.root'), key=self.tree_name)


class BootstrapTraining(b2luigi.Task):
    """Start a training with a resampled train sample. See also `Training`.

    Parameters:
        random_seed (int): random seed of the resampled train sample
        off_res_files (list): List with paths to off-res. files
        tree_name (str): name of the tree in the root file
        training_variables (list): list of training variables used for training
        training_parameters (dict): train- and test size,
            the following BDT hyper-parameters (optional): "nTrees",
            "shrinkage" and "nLevels".

    Output:
        bdt.xml
    """
    random_seed = b2luigi.IntParameter()
    off_res_files = b2luigi.ListParameter(hashed=True)
    tree_name = b2luigi.ListParameter()
    training_variables = b2luigi.ListParameter(hashed=True)
    training_parameters = b2luigi.DictParameter(hashed=True)

    def requires(self):
        train_size = self.training_parameters["train_size"]
        test_size = self.training_parameters["test_size"]

        for ntuple_file in self.off_res_files:
            yield self.clone(
                Resample,
                ntuple_file=ntuple_file,
                train_size=train_size,
                test_size=test_size,
                random_seed=self.random_seed)

    def output(self):
        yield self.add_to_output('bdt.xml')

    def run(self):
        Training.run(self)


############
# Validation
############

@b2luigi.inherits(BootstrapTraining)
class BootstrapValidationExpert(b2luigi.Task):
    """Apply the BDT trained on the resampled train sample to the test sample.

    Parameters:
        random_seed, off_res_files, tree_name, training_variables,
        training_parameters: see BootstrapTraining.
    """
    def requires(self):
        yield self.clone_parent()

        test_size = self.training_parameters["test_size"]
        train_size = self.training_parameters["train_size"]
        for off_res_file in self.off_res_files:
            yield self.clone(
                SplitSample,
                ntuple_file=off_res_file,
                train_size=train_size,
                test_size=test_size
            )

    def output(self):
        yield self.add_to_output('validation_expert.root')

    def run(self):
        ValidationExpert.run(self)


@b2luigi.requires(BootstrapValidationExpert)
class BootstrapValidationReweighting(b2luigi.Task):
    """Reweight the test samples from the BootstrapExpert.

    Parameters:
        random_seed, off_res_files, tree_name, training_variables,
        training_parameters: see BootstrapTraining.
    """

    def output(self):
        yield self.add_to_output('validation_weights.root')

    def run(self):
        ValidationReweighting.run(self)


###########################
# Reweighting on-res. files
###########################

@b2luigi.requires(BootstrapTraining)
class BootstrapExpert(b2luigi.Task):
    """Apply the BDT trained on the resampled train sample to the
    on-res. Continuum MC.

    Parameters:
        random_seed, off_res_files, tree_name, training_variables,
        training_parameters: see BootstrapTraining.
    """
    on_res_files = b2luigi.ListParameter(hashed=True)
    queue = "sx"

    def output(self):
        yield self.add_to_output('expert.root')

    def run(self):
        Expert.run(self)


@b2luigi.inherits(BootstrapExpert)
class BootstrapReweighting(b2luigi.Task):
    """Reweight the on-res. MC samples. Calculate weights from
    the BootstrapExpert.

    Parameters:
        random_seed, off_res_files, tree_name, training_variables,
        training_parameters: see BootstrapTraining.
    """

    def requires(self):
        yield self.clone_parent()
        yield self.clone(
            BootstrapValidationReweighting,
            off_res_files=self.off_res_files,
            training_variables=self.training_variables,
            **self.training_parameters)

    def output(self):
        yield self.add_to_output("weights.root")

    def run(self):
        Reweighting.run(self)


############
# Delegation
############

class DelegateBootstrapping(b2luigi.Task):
    """Delegate the set of trainings and the reweighting of the test- and
    on-res. MC samples. The number_of_training parameter has to be set in the
    parameter file.

    The different sets of weights are listed in two different output files.

    Parameters:
        name, parameter_file: analogues to DelegateReweighting.

    Output:
        bootstrap_results.json
        bootstrap_validation_results.json
    """
    name = b2luigi.Parameter()
    parameter_file = b2luigi.Parameter(significant=False)

    def requires(self):
        with open(self.parameter_file) as parameter_file:
            parameters = yaml.load(parameter_file)

        off_res_files = parameters.get("off_res_files")
        on_res_files = parameters.get("on_res_files")
        training_parameters = parameters.get("training_parameters")
        number_of_trainings = parameters["number_of_trainings"]

        # Split Sample

        for off_res_file in off_res_files:
            # Split Sample
            yield self.clone(
                SplitSample,
                ntuple_file=off_res_file,
                tree_name=parameters["tree_name"],
                train_size=training_parameters.get("train_size"),
                test_size=training_parameters.get("test_size"),
            )

        for i in range(number_of_trainings):
            yield self.clone(
                BootstrapValidationReweighting,
                off_res_files=off_res_files,
                tree_name=parameters["tree_name"],
                training_variables=parameters["training_variables"],
                training_parameters=training_parameters,
                random_seed=i
                )

            if len(on_res_files) != 0:
                # BootstrapReweighting
                yield self.clone(
                    BootstrapReweighting,
                    off_res_files=off_res_files,
                    on_res_files=on_res_files,
                    tree_name=parameters["tree_name"],
                    training_variables=parameters["training_variables"],
                    training_parameters=training_parameters,
                    random_seed=i
                    )

        if len(on_res_files) == 0:
            print(
                "No on-resonance files are given."
                "Only test samples will be reweighted.")

    def output(self):
        yield self.add_to_output('bootstrap_results.json')

    def run(self):
        # on-res. weights
        weights_list = self.get_input_file_names('weights.root')

        # test sample weights
        test_samples = self.get_input_file_names("test.root")
        validation_weights_list = self.get_input_file_names(
            "validation_weights.root")

        bootstap_results = {
            "weights_list": weights_list,
            "test_samples": test_samples,
            "validation_weights_list": validation_weights_list,
        }

        with open(self.get_output_file_name('bootstrap_results.json'), 'w')\
                as f:
            json.dump(bootstap_results, f)
