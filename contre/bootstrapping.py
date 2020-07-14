import yaml
import b2luigi
import root_pandas
from sklearn.utils import resample
from contre.training import SplitSample, Training
from contre.reweighting import Expert, Reweighting
from contre.validation import ValidationExpert, ValidationReweighting


@b2luigi.requires(SplitSample)
class Resample(b2luigi.Task):
    random_seed = b2luigi.IntParameter()

    def output(self):
        yield self.add_to_output("train.root")

    def run(self):
        df = root_pandas.read_root(
            *self.get_input_file_names('train.root'),
            key='ntuple')

        # resample
        resampled_df = resample(df, random_state=self.random_seed)

        # store to root
        root_pandas.to_root(
            resampled_df,
            self.get_output_file_name('train.root'), key=self.tree_name)


class BootstrapTraining(b2luigi.Task):
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
                test_size=test_size)

    def output(self):
        yield self.add_to_output('bdt.xml')

    def run(self):
        Training.run()


############
# Validation
############

@b2luigi.requires(BootstrapTraining)
class BootstrapValidationExpert(b2luigi.Task):
    """Apply the BDT trained on the resampled train sample to the test sample.
    """

    def output(self):
        yield self.add_to_output('validation_expert.root')

    def run(self):
        ValidationExpert.run()


@b2luigi.requires(BootstrapValidationExpert)
class BootstrapValidationReweighting(b2luigi.Task):

    def output(self):
        yield self.add_to_output('validation_weights.root')

    def run(self):
        ValidationReweighting.run()


###########################
# Reweighting on-res. files
###########################

@b2luigi.requires(BootstrapTraining)
class BootstrapExpert(b2luigi.Task):
    """Apply the BDT trained on the resampled train sample to the
    on-res. Continuum MC."""
    on_res_files = b2luigi.ListParameter(hashed=True)
    queue = "sx"

    def output(self):
        yield self.add_to_output('expert.root')

    def run(self):
        Expert.run()


@b2luigi.inherits(BootstrapExpert)
class BootstrapReweighting(b2luigi.Task):
    """Caclulate weights from the BootstrapExpert."""

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
        Reweighting.run()


############
# Delegation
############

class DelegateBootstrapping(b2luigi.Task):
    """Create filelist with paths for the experts.

    Delegate BootstrapTraining and BootstrapExpert with resampled data train
    samples for a given number of resampled samples.

    Parameters:
        name, parameter_file: analouges to DelegateReweighting

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

        for i in range(number_of_trainings):
            for off_res_file in off_res_files:
                # Resample
                yield self.clone(
                    Resample,
                    ntuple_file=off_res_file,
                    tree_name=parameters["tree_name"],
                    train_size=training_parameters.get("train_size"),
                    test_size=training_parameters.get("test_size"),
                    random_seed=i,
                )

            # BootstrapTraining
            yield self.clone(
                BootstrapTraining,
                off_res_files=off_res_files,
                training_variables=parameters["training_variables"],
                tree_name=parameters["tree_name"],
                training_parameters=training_parameters,
                random_seed=i,
                )

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
        expert_list = self.get_input_file_names('expert.root')
        with open(self.get_output_file_name('expert_file_list.txt'), 'w') as f:
            f.write('#root experts\n')
            for expert in expert_list:
                f.write(expert + '\n')
