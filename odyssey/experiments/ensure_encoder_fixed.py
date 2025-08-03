
from odyssey.train.train import train
from odyssey.src.configurations import *

"""
The purpose of this file is to test that during stage 2 training, the FSQ encoder is actually fixed.

If the FSQ encoder parameters update during stage 2 training, then this test will fail.
"""

# Create the consensus block configuration
consensus_block_cfg = SelfConsensusConfig(
    consensus_num_iterations=1,
    consensus_connectivity_type="local_window",
    consensus_w=2,
    consensus_r=8,
    consensus_edge_hidden_dim=12
)

cross_consensus_block_cfg = CrossConsensusConfig(
    consensus_num_iterations=1,
    consensus_connectivity_type="local_window",
    consensus_w=2,
    consensus_r=8,
    consensus_edge_hidden_dim=12
)

# Create encoder configuration
encoder_cfg = FSQEncoderConfig(
    latent_dim=32,
    fsq_levels="7x5x5x5x5",
    d_model=128,
    n_heads=1,
    n_layers=3,
    max_len=2048,
    dropout=0.1,
    ff_mult=4,
    first_block_cfg=consensus_block_cfg,
    context_cfg=cross_consensus_block_cfg
)

# Create decoder configuration  
decoder_cfg = FSQDecoderConfig(
    latent_dim=32,
    fsq_levels="7x5x5x5x5", 
    d_model=128,
    n_heads=1,
    n_layers=3,
    max_len=2048,
    dropout=0.1,
    ff_mult=4,
    first_block_cfg=consensus_block_cfg,
    context_cfg=cross_consensus_block_cfg
)

# Stage 1 model configuration
stage_1_model_cfg = AutoencoderConfig(
    style="stage_1",
    autoencoder_path=None,
    reference_model_seed=42,
    vocab_domains_path="/workspace/demo/Odyssey/odyssey/train/vocab_domains.txt",
    vocab_orthologous_groups_path="/workspace/demo/Odyssey/odyssey/train/vocab_orthologous_groups.txt",
    vocab_semantic_description_path="/workspace/demo/Odyssey/odyssey/train/vocab_semantic_descriptions.txt",
    max_domains_per_residue=4,
    max_len_orthologous_groups=512,
    max_len_semantic_description=128,
    encoder_cfg=encoder_cfg,
    decoder_cfg=decoder_cfg,

)

# Stage 2 model configuration - uses pretrained stage 1 encoder
stage_2_model_cfg = AutoencoderConfig(
    style="stage_2",
    autoencoder_path="/workspace/demo/Odyssey/checkpoints/fsq/fsq_stage_1_config/fsq_stage_1_config_000/checkpoint_step_612.pt",
    reference_model_seed=42,
    vocab_domains_path="/workspace/demo/Odyssey/odyssey/train/vocab_domains.txt",
    vocab_orthologous_groups_path="/workspace/demo/Odyssey/odyssey/train/vocab_orthologous_groups.txt",
    vocab_semantic_description_path="/workspace/demo/Odyssey/odyssey/train/vocab_semantic_descriptions.txt",
    max_domains_per_residue=4,
    max_len_orthologous_groups=512,
    max_len_semantic_description=128,
    encoder_cfg=encoder_cfg,
    decoder_cfg=decoder_cfg,
)

# Call post_init for model configs
stage_1_model_cfg.__post_init__()
stage_2_model_cfg.__post_init__()

# Loss and mask configurations
loss_cfg = KabschRMSDLossConfig()

mask_cfg = DiffusionMaskConfig(
    corruption_mode="uniform",
    noise_schedule="uniform",
    sigma_min=0.31,
    sigma_max=5.68,
    num_timesteps=100
)

no_mask_cfg = NoMaskConfig()

# Training configurations
stage_1_train_cfg = TrainingConfig(
    loss_config=loss_cfg,
    mask_config=mask_cfg,
    batch_size=4,
    max_epochs=1,
    checkpoint_freq=None,
    max_steps_val=None,
    optim_schedule_config=FlatSchedulerConfig(learning_rate=0.1),
    data_dir="/workspace/demo/Odyssey/sample_data/100.csv",
    checkpoint_dir="/workspace/demo/Odyssey/checkpoints/fsq/test/",
)

stage_2_train_cfg = TrainingConfig(
    loss_config=loss_cfg,
    mask_config=no_mask_cfg,
    batch_size=4,
    max_epochs=1,
    optim_schedule_config=FlatSchedulerConfig(learning_rate=0.1),
    data_dir="/workspace/demo/Odyssey/sample_data/100.csv",
    checkpoint_dir="/workspace/demo/Odyssey/checkpoints/fsq/test/",
)

# Call post_init for training configs
stage_1_train_cfg.__post_init__()
stage_2_train_cfg.__post_init__()

# Global variables to track parameter changes
last_encoder_first_layer_params = None
last_decoder_first_layer_params = None

def callback(ret):
    global last_encoder_first_layer_params, last_decoder_first_layer_params
    import torch

    # Get first parameter from encoder
    for p in ret['model'].encoder.parameters():
        new_encoder_first_layer_params = p.data.clone()
        break

    # Get first parameter from decoder
    for p in ret['model'].decoder.parameters():
        new_decoder_first_layer_params = p.data.clone()
        break

    if last_encoder_first_layer_params is not None:
        enc_diff = torch.abs(last_encoder_first_layer_params - new_encoder_first_layer_params)
        dec_diff = torch.abs(last_decoder_first_layer_params - new_decoder_first_layer_params)

        print(f"Encoder mean absolute difference: {enc_diff.mean()}")
        print(f"Decoder mean absolute difference: {dec_diff.mean()}")

        # Assert encoder parameters haven't changed (should be frozen in stage 2)
        assert torch.allclose(enc_diff, torch.zeros_like(enc_diff)), "Encoder parameters changed during stage 2 training!"

    last_encoder_first_layer_params = new_encoder_first_layer_params
    last_decoder_first_layer_params = new_decoder_first_layer_params

def run_test():
    try:
        # Run stage 2 training and observe that encoder is fixed
        train([stage_2_model_cfg], [stage_2_train_cfg], callback=callback)
    except AssertionError as e:
        print("Test Failure: Encoder parameters changed during stage 2 training!")
        print(f"Error: {e}")
        return False

    print("Test Success: Encoder remained fixed during stage 2 training!")
    return True

if __name__ == "__main__":
    run_test()