"""
Enhanced engine functions that support passing images to the criterion for color contrast loss.
"""

import torch
from rfdetr.engine import train_one_epoch as train_one_epoch_original
from rfdetr.engine import evaluate as evaluate_original
from rfdetr.models.enhanced_criterion import EnhancedSetCriterion


def is_enhanced_criterion(criterion):
    """Check if criterion is an EnhancedSetCriterion."""
    if hasattr(criterion, 'module'):
        # Handle distributed training
        return isinstance(criterion.module, EnhancedSetCriterion)
    return isinstance(criterion, EnhancedSetCriterion)


def call_criterion_with_images(criterion, outputs, targets, samples):
    """
    Call criterion with images if it's an EnhancedSetCriterion.

    Args:
        criterion: The criterion module
        outputs: Model outputs
        targets: Ground truth targets
        samples: NestedTensor containing images
    """
    if is_enhanced_criterion(criterion):
        # Extract images from samples
        images = samples.tensors if hasattr(samples, 'tensors') else samples
        # Call with images
        if hasattr(criterion, 'module'):
            return criterion.module(outputs, targets, images=images)
        else:
            return criterion(outputs, targets, images=images)
    else:
        # Standard criterion call
        return criterion(outputs, targets)


# Note: For actual integration, you would need to modify rfdetr/engine.py
# to use call_criterion_with_images instead of calling criterion directly.
# This can be done by:
# 1. Importing call_criterion_with_images in engine.py
# 2. Replacing criterion(outputs, targets) with call_criterion_with_images(criterion, outputs, targets, samples)

# Example of how to patch the training loop:
def patch_train_one_epoch():
    """
    Returns a patched version of train_one_epoch that supports EnhancedSetCriterion.
    This is a reference implementation showing how to integrate the enhanced criterion.
    """
    import inspect
    import types

    # Get the source code of train_one_epoch
    source = inspect.getsource(train_one_epoch_original)

    # Replace criterion(outputs, targets) with call_criterion_with_images
    # This is just an example - in practice, you'd modify the engine.py file directly

    print("To integrate enhanced criterion:")
    print("1. In rfdetr/engine.py, import: from rfdetr.engine_enhanced import call_criterion_with_images")
    print("2. Replace 'loss_dict = criterion(outputs, new_targets)' with:")
    print("   'loss_dict = call_criterion_with_images(criterion, outputs, new_targets, new_samples)'")
    print("3. Replace 'loss_dict = criterion(outputs, targets)' in evaluate function with:")
    print("   'loss_dict = call_criterion_with_images(criterion, outputs, targets, samples)'")


if __name__ == "__main__":
    patch_train_one_epoch()
