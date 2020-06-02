import root_pandas
import basf2_mva
import b2luigi
from sklearn.model_selection import train_test_split


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
    train = train[train["__candidate__"] == 0]
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
    tree_name = b2luigi.Parameter()
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
            train, self.get_output_file_name('train.root'), key=self.tree_name)
        root_pandas.to_root(
            test, self.get_output_file_name('test.root'), key=self.tree_name)


class Training(b2luigi.Task):
    """Train bdt on train samples and save bdt_weightfile to `bdt.xml`.

    Parameters:
        off_res_files (list): list of off res. files to be used for training,
        tree_name (str): name of the tree in the root_file,
        training_variables (list): variables used for training,
        training_parameters (dict): train- and test size, BDT hypermaremeters.
    """
    off_res_files = b2luigi.ListParameter(hashed=True)
    tree_name = b2luigi.ListParameter()
    training_variables = b2luigi.ListParameter(hashed=True)
    training_parameters = b2luigi.DictParameter(hashed=True)
    queue = "sx"

    def requires(self):
        train_size = self.training_parameters["train_size"]
        test_size = self.training_parameters["test_size"]

        for ntuple_file in self.off_res_files:
            yield self.clone(
                SplitSample,
                ntuple_file=ntuple_file,
                train_size=train_size,
                test_size=test_size)

    def output(self):
        yield self.add_to_output('bdt.xml')

    def run(self):
        bdt = self.get_output_file_name('bdt.xml')
        train_samples = self.get_input_file_names('train.root')

        # bdt options
        general_options = basf2_mva.GeneralOptions()
        general_options.m_datafiles = basf2_mva.vector(*train_samples)
        general_options.m_identifier = bdt
        general_options.m_treename = self.tree_name
        general_options.m_variables = basf2_mva.vector(
            *self.training_variables)
        general_options.m_target_variable = "EventType"

        fastbdt_options = basf2_mva.FastBDTOptions()
        # TODO: Implement More Fast BDT Options!

        # teacher
        basf2_mva.teacher(general_options, fastbdt_options)
