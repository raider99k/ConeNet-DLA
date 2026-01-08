
import torch
import torch.nn as nn

try:
    from apex.contrib.sparsity import ASP
    HAS_APEX_SPARSITY = True
except ImportError:
    HAS_APEX_SPARSITY = False
    print("WARNING: apex.contrib.sparsity (ASP) not found. Sparsity features will be disabled.")

def apply_24_sparsity(model, optimizer, whitelist=None):
    """
    Applies 2:4 structured sparsity to the model.
    whitelist: List of module names or modules to prune.
    """
    if not HAS_APEX_SPARSITY:
        return
    
    print("Applying 2:4 Structured Sparsity...")
    
    # Initialize ASP
    asp = ASP()
    
    # If whitelist is provided, we only prune those modules.
    # ASP usually prunes all supported layers. 
    # To restrict, we can either use the ASP.init_model_for_pruning with a filter.
    
    if whitelist:
        # We can implement a filter for ASP
        def whitelist_filter(name, module):
            # Check if name starts with any of the whitelist prefixes
            return any(name.startswith(w) for w in whitelist)
        
        asp.init_model_for_pruning(model, mask_calculator="24", verbosity=2, whitelist=whitelist_filter)
    else:
        asp.init_model_for_pruning(model, mask_calculator="24", verbosity=2)
        
    asp.compute_sparse_masks()
    
    # Re-init optimizer if needed or wrap it
    # ASP.init_optimizer_for_pruning(optimizer) -> Depends on version
    # Modern ASP handles it within the loop or via a call.
    
    print("Sparsity masks computed.")
    return asp

def enable_sparsity(asp):
    if HAS_APEX_SPARSITY and asp:
        asp.enable_sparsity()

def disable_sparsity(asp):
    if HAS_APEX_SPARSITY and asp:
        asp.disable_sparsity()
