from collections import namedtuple

niv_remote = '/home/tomergolany/ecg/'
local_base = '/Users/tomer.golany/PycharmProjects/'
nlp_base = '/home/tomer.golany@st.technion.ac.il/'
tomer_remote = '/home/tomer/tomer/'
niv2_remote = '/home/tomer/'
niv3_remote = '/media/drive/'
yochai_remote = '/home/yochaiz/tomergolany/'
google_remote = '/usr/local/google/home/tomergolany/'
colab_remote = '/content/'
niv_server = '/home/tomergolany/'
base = local_base

ECGTrainConfig = namedtuple('ECGTrainConfig',
                            'num_epochs batch_size lr weighted_loss weighted_sampling device')

GeneratorAdditionalDataConfig = namedtuple('GeneratorAdditionalDataConfig', 'beat_type checkpoint_path num_examples_to_'
                                                                            'add gan_type')
