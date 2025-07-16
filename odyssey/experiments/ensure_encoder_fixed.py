
from odyssey.train.train import train
from odyssey.src.configurations import *

"""
The purpose of this file is to test that during stage 2 training, the FSQ encoder is actually fixed.

If the FSQ encoder parameters update during stage 2 training, then this test will fail.
"""

block_cfg = SelfConsensusConfig(
    consensus_num_iterations=1,
    consensus_connectivity_type="local_window",
    consensus_w=2,
    consensus_r=8,
    consensus_edge_hidden_dim=12
)

stage_1_model_cfg = FSQConfig(
    style= "stage_1",  # Options: "stage_1", "stage_2", "mlm", "discrete_diffusion"
    d_model= 128,
    n_heads=1,
    n_layers= 3,
    max_len= 2048,
    dropout= 0.1,
    ff_mult= 4,
    reference_model_seed= 42,
    latent_dim= 32,
    fsq_levels= "7x5x5x5x5",
    fsq_encoder_path= None,  # Required for stage_2
    first_block_cfg=block_cfg,
    context_cfg= None,
)

stage_2_model_cfg = FSQConfig(
    style= "stage_2",  # Options= "stage_1", "stage_2", "mlm", "discrete_diffusion"
    d_model= 128,
    n_heads= 1,
    n_layers= 3,
    max_len= 2048,
    dropout= 0.1,
    ff_mult= 4,
    reference_model_seed= 42,
    latent_dim= 32,
    fsq_levels= "7x5x5x5x5",
    fsq_encoder_path= "/workspace/demo/Odyssey/checkpoints/fsq/SC_stage_1_simple_model.pt",  # Required for stage_2
    first_block_cfg=block_cfg,
    context_cfg=None,
)

loss_cfg = KabschRMSDLossConfig()

mask_cfg = DiffusionMaskConfig(
    noise_schedule="uniform",
    sigma_min=0.31,
    sigma_max=5.68,
    num_timesteps=100
)

no_mask_cfg = NoMaskConfig()

stage_1_train_cfg = TrainingConfig(
    loss_config=loss_cfg,
    mask_config=mask_cfg,
    batch_size=4,
    max_epochs=3,
    learning_rate=0.00001,
    data_dir="/workspace/demo/Odyssey/sample_data/100.csv",
    checkpoint_dir="/workspace/demo/Odyssey/checkpoints/tmp",
)

stage_2_train_cfg = TrainingConfig(
    loss_config=loss_cfg,
    mask_config=no_mask_cfg,
    batch_size=4,
    max_epochs=4,
    learning_rate=0.1,
    data_dir="/workspace/demo/Odyssey/sample_data/100.csv",
    checkpoint_dir="/workspace/demo/Odyssey/checkpoints/tmp",
)



last_encoder_first_layer_params = None
last_decoder_first_layer_params = None
def callback(ret):
    global last_encoder_first_layer_params, last_decoder_first_layer_params

    for p in ret['model'].encoder.parameters():
        new_encoder_first_layer_params = p.data.clone()
        break

    for p in ret['model'].decoder.parameters():
        new_decoder_first_layer_params = p.data.clone()
        break

    if last_encoder_first_layer_params is not None:
        enc_diff = torch.abs(last_encoder_first_layer_params - new_encoder_first_layer_params)
        dec_diff = torch.abs(last_decoder_first_layer_params - new_decoder_first_layer_params)

        print(f"Encoder mean absolute difference: {enc_diff.mean()}")
        print(f"Decoder mean absolute difference: {dec_diff.mean()}")

        assert torch.allclose(enc_diff, torch.zeros_like(enc_diff))


    last_encoder_first_layer_params = new_encoder_first_layer_params
    last_decoder_first_layer_params = new_decoder_first_layer_params

    


# Run Stage 1 training.  Watch both encoder and decoder parameters update:
# train(_stage_1_model_cfg, _stage_1_train_cfg, callback=callback)

# Run stage 2 training and observe that encoder is fixed:
# train(_stage_2_model_cfg, _stage_2_train_cfg, callback=callback)



def run_test():
    try:
        train(stage_2_model_cfg, stage_2_train_cfg, callback=callback)
    except AssertionError as e:
        print("Test Failure.")
        return False

    print("Test Success.")
    return True


if __name__ == "__main__":
    run_test()