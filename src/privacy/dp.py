from __future__ import annotations

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


def make_private_with_opacus(model, optimizer, train_loader, noise_multiplier: float, max_grad_norm: float):
    optimizer_cls = optimizer.__class__
    optimizer_defaults = dict(optimizer.defaults)

    model = ModuleValidator.fix(model)
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        raise RuntimeError(f"Opacus module validation failed: {errors}")

    optimizer = optimizer_cls(model.parameters(), **optimizer_defaults)

    privacy_engine = PrivacyEngine()
    model, optimizer, private_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    return model, optimizer, private_loader, privacy_engine