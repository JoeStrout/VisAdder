## Mar 3, 2026

I'm starting my work on LLM visualization with a tiny (36 free parameters!) transformer from [AdderBoard](https://github.com/anadim/AdderBoard?).  That [code](https://gist.github.com/alexlitz/0d5efbccf443fb0e8136b8f5bd85140a) doesn't have a repo of its own, but I've copied it here under litz_adder (named for its author, Alex Litzenberger).

The code runs and self-tests successfully in my default Python environment.  Ah, but my default environment doesn't have `torchinfo`... I'm going to create a new `vislm` micromamba environment, and install pytorch and related tools, along with probably Raylib bindings, therein.

Next issue: the TinyAdder class Alex wrote doesn't inherit from nn.Module, so we can't use most of the standard visualization tools (including `torchinfo`) on it.  So I'm making (in tinyadder_module.py) an nn.Module version of the same architecture.

Annoyingly, it seems that on this platform (MacOS 13.6), micromamba doesn't have a recent version of pytorch, and so I'm having to downgrade numpy to 1.x as well.

OK, that's all sorted out; torchinfo reports:
```
==========================================================================================
Layer (type:depth-idx)                   Param #                   Output Shape
==========================================================================================
TinyAdderModule                          --                        [1, 23]
├─SparseEmbedding: 1-1                   --                        [1, 23, 5]
├─Layer0Attention: 1-2                   6                         [1, 23, 5]
├─Layer0FFN: 1-3                         12                        [1, 23, 11]
├─Layer1Attention: 1-4                   2                         [1, 23, 1]
├─Layer1FFN: 1-5                         3                         [1, 23, 10]
==========================================================================================
Total params: 23
```

Pytorch counts the parameters slightly differently; it's not counting the buffer used as a  lookup table for the embedding module, while the contest does count it.

Next: installing `torchview` to try to generate a model diagram.  Made draw_graph.py, which generated tinnyadder_graph.png.  This visualizes the computations going on in the `forward` method, with green boxes denoting actual `nn.Module` submodules, and white boxes representing other operations.  23 (the second dimension in all of these tensors) is the context size.  But this is not fixed; the model handles variable-length sequences dynamically.  The context actually grows as the model cranks away, adding one token at a time. 

Running the script with `ipython -i` lets me inspect some useful things, like the TOKENS list: `['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '<bos>', '<eos>', '+']`.  This is how tokens are mapped to numbers in the tensors.

Because the model syzes dynamically, I'm going to pad with fewer digits, which will make it easier to visualize (at least while my "visualization" involves dumping numbers to the terminal).

I'm now going through the model layer by layer.  I've gained an important insight: transformers use a "residual stream architecture" which means that every layer reads and writes to the same vector, so the width is determined by the maximum needs of any layer.  Claude explains:

 The residual stream is a fixed-width bus that all layers share. In GPT-2 for example, d_model=768, and different attention heads and FFN layers effectively "claim" different subspaces to write their results into, while other components leave those dimensions alone.                                                                                                     
                                                                                                                                               
  The difference is just one of scale and legibility. In a trained 768-dim model, the subspaces overlap, interfere, and are discovered through gradient descent — so you can't easily point to "dim 347 is the count channel." Here, with only 5 dimensions, the author has hand-assigned each dim a clear purpose, making the residual stream architecture visible in a way that's normally hidden behind billions of parameters.

  So the "empty" dims in the embedding and the "silent" heads aren't a shortcut — they're the same structural pattern as any transformer, just small enough to see.

But note that a standard Transformer, when it has a FFN layer, expands (typically to 4x the model width) internally, and then projects back down to the standard residual width.  This TinyAdder module doesn't do that; instead, it widens the residual stream itself, because later layers need those extra dims.

Also unusual in this model: the author has fixed Q=K=0, i.e., zero weights instead of learned linear projections of each position's state vector.  Q=K=0 is a degenerate case that produces uniform attention: every position looks equally at everything for it.  So the attention mechanism (with the causal mask, which makes each position look only at the positions before it) reduces to a simple running average.  In most applications that would be a very dumb model indeed, but in this case it's very handy for summing up the place-value contributions, and figuring out what is left to represent as the next number.

Worked with Claude to write up a [step-by-step walk-through](understanding_tinyadder.md) of how this thing works.  It's beautiful and skanky all at once.

Now I'm developing vis_adder.py, which draws the output of each layer as a labeled heatmap, along with some notes about what's going on.  This is all done in Raylib, and you can zoom in/out to shift between overview and detail.  Also, you can change the inputs, and step forward and backward through generation of the output.

