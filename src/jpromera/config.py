"""Hardcoded, fully-resolved Promera model config (copied from
promera/model/config.yaml with interpolations resolved) so the model builds
without OmegaConf or an external yaml path."""


class AttrDict(dict):
    """dict with attribute access + ** unpacking (an OmegaConf-free config node)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v

    def __delattr__(self, k):
        del self[k]


def _wrap(x):
    if isinstance(x, dict):
        return AttrDict({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def get_config():
    """Return the model config as a fresh AttrDict tree (safe to mutate)."""
    return _wrap(CONFIG)


CONFIG = {   'set_float32_matmul_precision': 'high',
    'trainer': {   'accelerator': 'gpu',
                   'devices': 1,
                   'precision': 32,
                   'logger': False,
                   'enable_progress_bar': True,
                   'strategy': 'auto'},
    'data': {'batch': 1, 'workers': 1},
    'feature': {   'pad_to_max_tokens': True,
                   'pad_to_max_atoms': True,
                   'pad_to_max_seqs': True,
                   'min_dist': 2.0,
                   'max_dist': 22.0,
                   'num_bins': 64,
                   'atoms_per_window_queries': 32,
                   'mask_std_feats': True},
    'model': {   '_target_': 'promera.model.model.PromeraModel',
                 'disto_embed': True,
                 'subsample_msa_per_recycle': True,
                 'center_of_mass_loss': True,
                 'center_of_mass_loss_weight': 0.001,
                 'msas_per_trunk_iter': 1024,
                 'dims': {   'atom_s': 128,
                             'atom_z': 16,
                             'token_s': 384,
                             'token_z': 128,
                             'atom_feature_dim': 389,
                             'atoms_per_window_queries': 32,
                             'atoms_per_window_keys': 128},
                 'num_bins': 64,
                 'ema': True,
                 'val_ema': True,
                 'ema_decay': 0.999,
                 'min_dist': 2.0,
                 'max_dist': 22.0,
                 'input_embedder': {   'dims': {   'atom_s': 128,
                                                   'atom_z': 16,
                                                   'token_s': 384,
                                                   'token_z': 128,
                                                   'atom_feature_dim': 389,
                                                   'atoms_per_window_queries': 32,
                                                   'atoms_per_window_keys': 128},
                                       'atom_encoder_depth': 0,
                                       'atom_encoder_heads': 4,
                                       'feature': {   'pad_to_max_tokens': True,
                                                      'pad_to_max_atoms': True,
                                                      'pad_to_max_seqs': True,
                                                      'min_dist': 2.0,
                                                      'max_dist': 22.0,
                                                      'num_bins': 64,
                                                      'atoms_per_window_queries': 32,
                                                      'mask_std_feats': True}},
                 'msa_args': {   'msa_s': 64,
                                 'msa_blocks': 4,
                                 'msa_dropout': 0.15,
                                 'z_dropout': 0.25,
                                 'pairwise_head_width': 32,
                                 'pairwise_num_heads': 4,
                                 'activation_checkpointing': True,
                                 'offload_to_cpu': False},
                 'pairformer_args': {   'dims': {   'atom_s': 128,
                                                    'atom_z': 16,
                                                    'token_s': 384,
                                                    'token_z': 128,
                                                    'atom_feature_dim': 389,
                                                    'atoms_per_window_queries': 32,
                                                    'atoms_per_window_keys': 128},
                                        'num_blocks': 48,
                                        'num_heads': 16,
                                        'dropout': 0.25,
                                        'pairwise_head_width': 32,
                                        'pairwise_num_heads': 4,
                                        'activation_checkpointing': True,
                                        'offload_to_cpu': False},
                 'has_structure_module': True,
                 'structure_module_args': {   'dims': {   'atom_s': 128,
                                                          'atom_z': 16,
                                                          'token_s': 384,
                                                          'token_z': 128,
                                                          'atom_feature_dim': 389,
                                                          'atoms_per_window_queries': 32,
                                                          'atoms_per_window_keys': 128},
                                              'compile_score': False,
                                              'diffusion': {'sigma_data': 16.0},
                                              'score_model_args': {   'sigma_data': 16,
                                                                      'dim_fourier': 256,
                                                                      'atom_encoder_depth': 3,
                                                                      'atom_encoder_heads': 4,
                                                                      'token_transformer_depth': 24,
                                                                      'token_transformer_heads': 16,
                                                                      'atom_decoder_depth': 3,
                                                                      'atom_decoder_heads': 4,
                                                                      'conditioning_transition_layers': 2,
                                                                      'activation_checkpointing': True,
                                                                      'offload_to_cpu': False,
                                                                      'has_alt_update': False}},
                 'has_confidence': True,
                 'confidence_module_args': {   'dims': {   'atom_s': 128,
                                                           'atom_z': 16,
                                                           'token_s': 384,
                                                           'token_z': 128,
                                                           'atom_feature_dim': 389,
                                                           'atoms_per_window_queries': 32,
                                                           'atoms_per_window_keys': 128},
                                               'num_dist_bins': 64,
                                               'max_dist': 22,
                                               'pairformer': {   'dims': {   'atom_s': 128,
                                                                             'atom_z': 16,
                                                                             'token_s': 384,
                                                                             'token_z': 128,
                                                                             'atom_feature_dim': 389,
                                                                             'atoms_per_window_queries': 32,
                                                                             'atoms_per_window_keys': 128},
                                                                 'num_blocks': 4,
                                                                 'num_heads': 16,
                                                                 'dropout': 0.25,
                                                                 'pairwise_head_width': 32,
                                                                 'pairwise_num_heads': 4,
                                                                 'activation_checkpointing': True,
                                                                 'offload_to_cpu': False}},
                 'has_contact_module': True,
                 'contact_module_args': {   'dims': {   'atom_s': 128,
                                                        'atom_z': 16,
                                                        'token_s': 384,
                                                        'token_z': 128,
                                                        'atom_feature_dim': 389,
                                                        'atoms_per_window_queries': 32,
                                                        'atoms_per_window_keys': 128},
                                            'num_dist_bins': 64,
                                            'max_dist': 22,
                                            'pairformer': {   'dims': {   'atom_s': 128,
                                                                          'atom_z': 16,
                                                                          'token_s': 384,
                                                                          'token_z': 128,
                                                                          'atom_feature_dim': 389,
                                                                          'atoms_per_window_queries': 32,
                                                                          'atoms_per_window_keys': 128},
                                                              'num_blocks': 4,
                                                              'num_heads': 16,
                                                              'dropout': 0.25,
                                                              'pairwise_head_width': 32,
                                                              'pairwise_num_heads': 4,
                                                              'activation_checkpointing': True,
                                                              'offload_to_cpu': False,
                                                              'no_update_s': True}}}}


# Default EDM diffusion sampling schedule (from promera/inference/cofolding.yaml).
DIFFUSION = {
    "sigma_min": 0.0004,
    "sigma_max": 160.0,
    "sigma_data": 16.0,
    "edm_churn": True,
    "rho": 7,
    "gamma_0": 0.8,
    "gamma_min": 1.0,
    "noise_scale": 1.003,
    "step_scale": 1.5,
}


# Token vocabulary size (tinyprot.feature._ntoks); fixed so core stays
# torch/tinyprot-free.
NTOKS = 34

# MSA subsampling: per recycle, the trunk draws this many MSA rows (matches
# PyTorch ``msas_per_trunk_iter`` with ``subsample_msa_per_recycle``).
MSAS_PER_TRUNK_ITER = CONFIG["model"]["msas_per_trunk_iter"]
