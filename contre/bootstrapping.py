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


class DelegateBootstrapping(b2luigi.Task):
    """Create filelist with paths for the experts.

    Delegate BootstrapTraining and BootstrapExpert with resampled data train
    samples for a given number of resampled samples.

    Parameters:
        experiment: experiment number
        train_size: size of the training sample
        test_size: size of the test sample
        r2_smaller_than (float): See SplitSapmle
        number_of_trainings: Number of trainings that should be performed for
            bootstraping.

    Output:
        Create a textfile with path to all experts (expert_file_list.txt)
    """
    experiment = b2luigi.IntParameter()
    train_size = b2luigi.FloatParameter()
    test_size = b2luigi.FloatParameter()
    r2_smaller_than = b2luigi.FloatParameter()
    number_of_trainings = b2luigi.IntParameter()

    def requires(self):
        for i in range(self.number_of_trainings):
            yield self.clone(
                BootstrapExpert,
                experiment=self.experiment,
                train_size=self.train_size,
                test_size=self.test_size,
                r2_smaller_than=self.r2_smaller_than,
                random_seed_resample=i)

    def output(self):
        yield self.add_to_output('expert_file_list.txt')  # does this work?

    def run(self):
        expert_list = self.get_input_file_names('expert.root')
        with open(self.get_output_file_name('expert_file_list.txt'), 'w') as f:
            f.write('#root experts\n')
            for expert in expert_list:
                f.write(expert + '\n')


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
        ValidaionExpert.run()


@b2luigi.requires(BootstrapValidationExpert)
class BootstrapValidationReweighting(b2luigi.Task):

    def output(self):
        yield self.add_to_output("")


class DelegateValidationBootstrapping(b2luigi.Task):
    """Delegate Reweighting of the test sample with the different trainings."""
