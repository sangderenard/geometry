# Auto-generated SSA helper functions

def ssa_helper_load(builder, operands, **kwargs):
    """Load a value from memory at a given address, optionally with alignment and volatility flags."""
    addr = operands[0]
    result = builder.fresh(dtype=kwargs.get("return_dtype"))
    builder.record(("Load", result, addr, kwargs.get("alignment"), kwargs.get("volatile")))
    return result

