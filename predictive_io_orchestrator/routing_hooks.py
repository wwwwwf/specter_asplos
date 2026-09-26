"""Connect model routers to PIO observations."""

def _configure_prefetch_routes(
    route_specs,
    hook_handle_list,
    prefetch_controller,
):
    prefetch_controller.bind_draft_routes(module for _, module in route_specs)


def set_hook_4_prefetch_noblock_qwenmoe(
    small_model,
    large_model,
    hook_handle_list,
    prefetch_controller,
):
    del large_model
    route_specs = [
        (layer_idx, layer.mlp)
        for layer_idx, layer in enumerate(
            small_model.model.model.layers
        )
    ]
    _configure_prefetch_routes(
        route_specs,
        hook_handle_list,
        prefetch_controller,
    )


def set_hook_4_prefetch_noblock_dsv2lite(
    small_model,
    large_model,
    hook_handle_list,
    prefetch_controller,
):
    del large_model
    route_specs = [
        (layer_idx, layer.mlp)
        for layer_idx, layer in enumerate(
            small_model.model.model.layers
        )
        if layer_idx != 0
    ]
    _configure_prefetch_routes(
        route_specs,
        hook_handle_list,
        prefetch_controller,
    )


def set_hook_4_prefetch_noblock_phimoe(
    small_model,
    large_model,
    hook_handle_list,
    prefetch_controller,
):
    del large_model
    route_specs = [
        (layer_idx, layer.block_sparse_moe)
        for layer_idx, layer in enumerate(
            small_model.model.model.layers
        )
    ]
    _configure_prefetch_routes(
        route_specs,
        hook_handle_list,
        prefetch_controller,
    )


