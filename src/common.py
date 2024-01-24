from os.path import join, abspath, dirname, pardir

BASE_DIR = abspath(join(dirname(__file__), pardir))# /localhome/kka151/Desktop/Research Coding/wfd-gan
confdir = join(BASE_DIR, 'conf.ini')
#outputdir = join(BASE_DIR, 'dump')
outputdir = join(BASE_DIR, 'outputs')
dModelDir = join(BASE_DIR, 'f_model')
LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"

# print(BASE_DIR)
# print(confdir)
# print(outputdir)
# Loss weight for gradient penalty
lambda_gp = 10

# for gan-glue
TRACE_SEP = '\t'

CELL_SIZE = 514
DUMMY_CODE = 888

# model name template
GENERATOR_NAME_REG = 'generator_seqlen(?P<seqlen>\d+)_cls(?P<cls>\d+)_latentdim(?P<latentdim>\d+).ckpt'
