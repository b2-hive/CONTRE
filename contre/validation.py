import json
import b2luigi
import root_pandas
import basf2_mva
from basf2 import conditions
from sklearn.model_selection import train_test_split
from contre.weights import get_weights


def split_sample(
        ntuple_file,
        train_size,
        test_size,
        random_seed=42):
    """Split rootfile and return dataframes."""
    df = root_pandas.read_root(ntuple_file)
    train, test = train_test_split(
        df,
        test_size=test_size,
        train_size=train_size,
        random_state=random_seed)
    return train, test


class SplitSample(b2luigi.Task):
    """Split a ntuple file to a training and test sample of given size.

    Shuffle events and create a random selected train and test sample.
    The function sklearn.utils.resample is used.
    Store samples as rootfiles (used by fastBDT).

    Parameters:
        ntuple_file (str): Path to the file
        train_size (float): between 0 and 1, size of train sample
        test_size (float): size of test sample
    """
    ntuple_file = b2luigi.Parameter(hashed=True)
    train_size = b2luigi.FloatParameter()
    test_size = b2luigi.FloatParameter()
    queue = "sx"

    def output(self):
        yield self.add_to_output('train.root')
        yield self.add_to_output('test.root')

    def run(self):
        train, test = split_sample(
            ntuple_file=self.ntuple_file,
            train_size=self.train_size,
            test_size=self.test_size)

        # Store as Rootfile
        root_pandas.to_root(
            train, self.get_output_file_name('train.root'), key='ntuple')
        root_pandas.to_root(
            test, self.get_output_file_name('test.root'), key='ntuple')


class ValidationTraining(b2luigi.Task):
    """Perform training for reweighting on train sample.

    Train bdt and save bdt_weightfile to `bdt.xml`. Apply BDT to test samples
    and save result as `expert.root`.

    Parameters:
        ntuple_files (list): files to be used for training
        training_variables (list): variables used for training
        train_size: Float, size of train sample
        test_size: Float, size of test sample
    """
    ntuple_files = b2luigi.ListParameter(hashed=True)
    training_variables = b2luigi.ListParameter(hashed=True)
    train_size = b2luigi.FloatParameter()
    test_size = b2luigi.FloatParameter()
    queue = "sx"

    def requires(self):
        for ntuple_file in self.ntuple_files:
            yield self.clone(
                SplitSample,
                ntuple_file=ntuple_file,
                train_size=self.train_size,
                test_size=self.test_size,)

    def output(self):
        yield self.add_to_output('bdt.xml')
        # yield self.add_to_output('expert.root')

    # @b2luigi.on_temporary_files
    def run(self):
        bdt = self.get_output_file_name('bdt.xml')
        # expert = self.get_output_file_name('expert.root')

        train_samples = self.get_input_file_names('train.root')
        # test_samples = self.get_input_file_names('test.root')

        # bdt options
        # conditions.testing_payloads = ['localdb/database.txt']
        general_options = basf2_mva.GeneralOptions()
        fastbdt_options = basf2_mva.FastBDTOptions()

        general_options.m_datafiles = basf2_mva.vector(*train_samples)
        general_options.m_identifier = bdt
        general_options.m_treename = "ntuple"
        general_options.m_variables = basf2_mva.vector(
            *self.training_variables)
        general_options.m_target_variable = "EventType"

        # teacher
        basf2_mva.teacher(general_options, fastbdt_options)

        # expert (apply bdt to test sample)
        # basf2_mva.expert(
        #     basf2_mva.vector(*bdt),
        #     basf2_mva.vector(*test_samples),
        #     'ntuple', expert)


@b2luigi.inherits(ValidationTraining)
class ValidationExpert(b2luigi.Task):
    """Apply trained BDT to test sample.

    Parameters: See ValidationTraining
    Output: expert.root
    """
    queue = "sx"

    def requires(self):
        yield self.clone_parent()
        # for test samples also require all split samples
        yield ValidationTraining.requires(self)

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
        for ntuple_file in parameters.get('off_res_files'):
            yield self.clone(
                SplitSample,
                ntuple_file=ntuple_file,
                **validation_parameters
            )
        yield self.clone(
            ValidationTraining,
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
            "train_samples": train_samples,
            "test_samples": test_samples,
            "bdt": bdt,
            "expert": expert,
            "weights": weights
        }

        with open(self.get_output_file_name(
                'validation_results.json'), 'w') as file:
            json.dump(results, file)
