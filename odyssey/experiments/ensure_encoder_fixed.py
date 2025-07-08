from test_cfg_objs import _stage_1_model_cfg, _stage_1_train_cfg, _stage_2_model_cfg, _stage_2_train_cfg
from odyssey.train.train_tensorized import train

# Run Stage 1 training.  Watch both encoder and decoder parameters update:
train(_stage_1_model_cfg, _stage_1_train_cfg, verbose=True)

# Run stage 2 training and observe that encoder is fixed:
train(_stage_2_model_cfg, _stage_2_train_cfg, verbose=True)
