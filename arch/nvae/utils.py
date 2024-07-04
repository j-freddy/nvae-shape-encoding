ID_TO_ARCH = {
    "default": {
        "num_groups_per_layer": [4, 2, 1],
        "is_layer_shared": [True, True, True],
        "initial_downsample_factor": 8,
    },
    "large-latent-skip": {
        "num_groups_per_layer": [4, 2, 2, 1, 1],
        "is_layer_shared": [True, False, True, False, True],
        "initial_downsample_factor": 2,
    }
}
