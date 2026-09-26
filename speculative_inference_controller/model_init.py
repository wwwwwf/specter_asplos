"""SIC model initialization with verified, aliased FP16 non-expert weights."""
import torch
from pathlib import Path
from dataclasses import replace


class HybridPrecisionModelInitializer:
    @staticmethod
    def prepare_fused_expert_precision(model, dtype=torch.float16):
        """Prepare expert floating state once, without changing packed INT4 data.

        Module.to() covers scales and registered buffers. The inherited fused
        experts also keep scratch tensors as ordinary attributes, so those must
        be converted explicitly before the first forward call.
        """
        prepared = []
        for name, fused in model.named_modules():
            if name.rsplit('.', 1)[-1] != 'fusedexperts':
                continue
            fused.to(dtype=dtype)
            plain_tensors = []
            for child_name, child in fused.named_modules():
                for attribute, tensor in list(vars(child).items()):
                    if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                        setattr(child, attribute, tensor.to(dtype=dtype))
                        plain_tensors.append('.'.join(filter(None, [child_name, attribute])))
            prepared.append({'module': name, 'dtype': str(dtype),
                             'plain_floating_tensors': plain_tensors})
        return prepared

    @staticmethod
    def resolve_model_paths(case):
        from initialization.checkpoint_layout import checkpoint_alias
        return replace(case, state_path=checkpoint_alias(case.state_path))

    @staticmethod
    def share_non_expert_parameters(draft, target):
        draft_model = draft.model
        target_params = dict(target.named_parameters())
        shared = []
        skipped = []
        conversions = []
        for name, parameter in list(draft_model.named_parameters()):
            if any(part in {'experts', 'fusedexperts'} for part in name.split('.')):
                continue
            target_parameter = target_params.get(name)
            if target_parameter is None:
                skipped.append(name)
                continue
            if parameter.shape != target_parameter.shape:
                raise ValueError(f'Non-expert layout mismatch: {name}')
            if target_parameter.dtype != torch.float16:
                raise ValueError(f'Expected FP16 target non-expert parameter: {name}: {target_parameter.dtype}')
            converted = parameter.to(target_parameter.dtype)
            difference = float((converted - target_parameter).abs().max())
            if parameter.dtype != target_parameter.dtype or difference:
                conversions.append({'name': name, 'draft_dtype': str(parameter.dtype), 'target_dtype': str(target_parameter.dtype), 'max_abs_difference': difference})
            parent_name, _, attribute = name.rpartition('.')
            parent = draft_model.get_submodule(parent_name) if parent_name else draft_model
            setattr(parent, attribute, target_parameter)
            shared.append(name)
        if skipped:
            raise ValueError(f'Unmatched non-expert parameters: {skipped}')
        if not shared:
            raise ValueError('No non-expert parameters were shared')
        # Expert scales/scratch and the shared backbone use FP16 directly.
        # Remove any old precision adapters if initialization is repeated on an
        # already-loaded wrapper; no forward hooks are installed here.
        for handle in getattr(draft, '_specter_precision_hooks', []):
            handle.remove()
        draft._specter_precision_hooks = []
        prepared = HybridPrecisionModelInitializer.prepare_fused_expert_precision(draft_model)
        torch.cuda.empty_cache()
        return {'shared_parameters': shared, 'shared_parameter_count': len(shared),
                'precision_conversions': conversions, 'fused_precision_adapter_count': 0,
                'fused_expert_precision': prepared}

    def load(self, case, device):
        from initialization.model_loader import load_models
        case = self.resolve_model_paths(case)
        tokenizer, draft, target = load_models(case, device)
        report = self.share_non_expert_parameters(draft, target)
        return tokenizer, draft, target, report
