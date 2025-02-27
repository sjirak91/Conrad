# Utility functions for NER processing

def compounding(start, stop, compound):
    """
    Yield compounding values, like spaCy's compounding function.
    
    Args:
        start: Start value
        stop: Stop value
        compound: Compound value (how quickly to increase)
    """
    curr = start
    while curr < stop:
        yield curr
        curr = curr * compound

def minibatch(items, size):
    """
    Divide a sequence into mini batches, like spaCy's minibatch function.
    
    Args:
        items: Items to batch
        size: Batch size or iterator of batch sizes
    """
    if isinstance(size, int):
        size = iter([size] * len(items))
    items = list(items)
    batch = []
    
    for size_i in size:
        batch.extend(items[:size_i])
        items = items[size_i:]
        if len(batch) > 0:
            yield batch
            batch = []
        if len(items) == 0:
            break 