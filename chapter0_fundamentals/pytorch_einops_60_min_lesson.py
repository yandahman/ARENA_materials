import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PyTorch & Einops: 60-Minute Tensor Fundamentals

    **For ARENA and mechanistic interpretability.** Work through the cells in order. **Predict shapes before running code.** Each solution starts collapsed. Click its **Solution** heading when you are ready to reveal it, and click again to hide it.

    | Section | Time |
    |---|---:|
    | Setup | 2 min |
    | Shapes | 8 min |
    | Indexing | 10 min |
    | Broadcasting | 12 min |
    | Reshape / stack / cat | 8 min |
    | Einops | 10 min |
    | Final challenge | 10 min |

    If `einops` is missing, activate your Python environment and run `python -m pip install einops` in a terminal.

    **Using this lesson in marimo:** names beginning with `_` are local to a cell, so examples and exercises can reuse them. Write each exercise's code in its starter cell, and run that cell to check your work.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0 · Setup (2 min)
    """)
    return


@app.cell
def _():
    import torch as t
    from einops import rearrange, repeat, reduce
    t.set_printoptions(precision=2)
    print("PyTorch", t.__version__)
    return rearrange, reduce, repeat, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Shapes (8 min)

    A tensor is a multidimensional array. The **meaning** of each dimension comes from the problem: `(batch, seq, d_model)` is a common convention, not something PyTorch knows intrinsically.

    - A scalar has shape `()`.
    - A vector has shape `(n,)`.
    - A matrix has shape `(rows, columns)`.
    - A batched matrix has shape `(batch, rows, columns)`.
    """)
    return


@app.cell
def _(t):
    scalar = t.tensor(5)
    vector = t.tensor([1, 2, 3])
    matrix = t.tensor([[1, 2, 3], [4, 5, 6]])
    batch = t.randn(4, 2, 3)
    for name, _x in [('scalar', scalar), ('vector', vector), ('matrix', matrix), ('batch', batch)]:
        print(name, _x.shape, 'ndim:', _x.ndim)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 1 — predict without running

    Given `_A=(5,10)`, `_B=(3,5,10)`, `_C=(2,3,4,5)`, predict `_A.ndim`, `_B.shape[-1]`, `_C.shape[1:]`, `_C.numel()`. If `_B` represents 3 images, which axis is the image axis?
    """)
    return


@app.cell
def _(t):
    _A = t.randn(5, 10)
    _B = t.randn(3, 5, 10)
    _C = t.randn(2, 3, 4, 5)
    # Write predictions here as comments, then uncomment to check:
    # 2, 10, (3,4,5), 120
    print(_A.ndim, _B.shape[-1], _C.shape[1:], _C.numel())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Solution 1 — click to reveal": mo.md(r"""
        **Solution 1:** `2`, `10`, `(3, 4, 5)`, `120`; images are axis 0.
        """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Indexing & slicing (10 min)

    A single integer index **removes** an axis; a slice (`:` or `0:1`) **keeps** it. `...` fills in unspecified axes. `None` / `unsqueeze` inserts a size-one axis.
    """)
    return


@app.cell
def _(t):
    _x = t.arange(24).reshape(2, 3, 4)
    print(_x)
    for label, y in [('x[0]', _x[0]), ('x[:,0]', _x[:, 0]), ('x[:,0:1]', _x[:, 0:1]), ('x[...,0]', _x[..., 0]), ('x[..., :2]', _x[..., :2]), ('x[:,None,:,:]', _x[:, None, :, :])]:
        print(label, tuple(y.shape))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2

    Let `_x` have shape `(8,12,64)` = `(batch, seq, embedding)`.

    Predict the shapes of `_x[0]`, `_x[:,0]`, `_x[:,:3]`, `_x[...,-1]`, `_x[:,None,:,:]`, `_x[:,:,None,:]`, `_x[0,:,:10]`.

    Then write expressions to (a) select first token embeddings for all batches, (b) select first 5 features for every token, (c) insert an axis between sequence and embedding.
    """)
    return


@app.cell
def _(t):
    _x = t.randn(8, 12, 64)
    # YOUR WORK
    print(_x[0].shape) #= (12,64)
    print(_x[:,0].shape) #= (8,64)
    print(_x[:,:3].shape) #= (8,3,64)
    print(_x[...,-1].shape) #= (8,12)
    print(_x[:,None,:,:].shape) #= (8,1,12,64)
    print(_x[:,:,None,:].shape) #= (8,12,1,64)
    print(_x[0,:,:10].shape) #= (12,10)
    _a = _x[:,0,:]
    _b = _x[...,:5]
    _c = _x[:,:,None,:]


    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Solution 2 — click to reveal": mo.md(r"""
        **Solution 2:** shapes `(12,64)`, `(8,64)`, `(8,3,64)`, `(8,12)`, `(8,1,12,64)`, `(8,12,1,64)`, `(12,10)`. Expressions: `_x[:,0,:]`, `_x[...,:5]`, `_x[:,:,None,:]`.
        """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Broadcasting (12 min)

    PyTorch aligns axes **from the right**. Each pair of axis lengths must match, or one must be 1 (missing leading axes behave as 1).

    ```
    (5, 1, 3)
    (1, 7, 3)
    ---------
    (5, 7, 3)
    ```

    A powerful pattern for all-pairs comparisons: `(n,d)` and `(m,d)` become `(n,1,d)` and `(1,m,d)`, yielding `(n,m,d)`.
    """)
    return


@app.cell
def _(t):
    _A = t.tensor([[1, 2, 3], [4, 5, 6]])
    _b = t.tensor([10, 20, 30])
    print(_A + _b)
    P = t.randn(5, 3)
    _Q = t.randn(7, 3)
    diff = P[:, None, :] - _Q[None, :, :]
    print('pairwise differences:', diff.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3

    1. Shape of `(4,1,8) + (1,6,8)`?
    2. Does `(5,3) + (3,5)` work?
    3. Given `_points=(10,2)`, compute a `(10,10)` matrix of pairwise Euclidean distances, **without loops**.
    """)
    return


@app.cell
def _(t):
    _points = t.randn(10, 2)
    # YOUR WORK
    # (4,6,8)
    # no
    _distances = ((_points[:,None,:]-_points[None,:,:])**2).sum(dim=-1).sqrt()
    assert _distances.shape == (10, 10)
    assert t.allclose(_distances, _distances.T, atol=1e-5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Solution 3 — click to reveal": mo.md(r"""
        **Solution 3:** `(4,6,8)`; no, 5 and 3 conflict; `_distances = t.linalg.vector_norm(_points[:,None,:] - _points[None,:,:], dim=-1)`.
        """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Reshape, stack, cat (8 min)

    - `reshape`: changes the shape while preserving element order (may create a copy).
    - `stack`: inserts a **new** axis; inputs must have equal shape.
    - `cat`: joins along an **existing** axis; other axes must match.
    - `unsqueeze`: inserts a size-one axis.
    """)
    return


@app.cell
def _(t):
    a = t.tensor([1, 2, 3])
    _b = t.tensor([4, 5, 6])
    print('stack dim 0:\n', t.stack((a, _b), dim=0))
    print('stack dim 1:\n', t.stack((a, _b), dim=1))
    _A = t.randn(2, 3)
    _B = t.randn(2, 3)
    print('cat 0:', t.cat((_A, _B), dim=0).shape, 'cat 1:', t.cat((_A, _B), dim=1).shape)
    print(_A)
    print(_B)
    print('cat 0:', t.cat((_A, _B), dim=0), '\ncat 1:', t.cat((_A, _B), dim=1))

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 4

    Given `_A` and `_B` each shaped `(5,3)`, produce `(2,5,3)`, `(5,2,3)`, `(10,3)`, and `(5,6)` using stack or cat.
    """)
    return


@app.cell
def _(t):
    _A = t.randn(5, 3)
    _B = t.randn(5, 3)
    # YOUR WORK
    print(t.stack((_A,_B), dim =0).shape)
    print(t.stack((_A,_B), dim =1).shape)
    print(t.cat((_A,_B), dim =0).shape)
    print(t.cat((_A,_B), dim =1).shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Solution 4 — click to reveal": mo.md(r"""
        **Solution 4:** `t.stack((_A,_B),0)`, `t.stack((_A,_B),1)`, `t.cat((_A,_B),0)`, `t.cat((_A,_B),1)`.
        """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Einops (10 min)

    Einops gives **names** to axes. Three essentials:

    - `rearrange`: reorder, flatten, or split axes.
    - `repeat`: add/repeat axes.
    - `reduce`: aggregate across axes that disappear from the output.

    `rearrange(x, "b s d -> b d s")` swaps sequence and feature axes. Parentheses group or split axes.
    """)
    return


@app.cell
def _(rearrange, reduce, repeat, t):
    _x = t.randn(32, 10, 768)
    print(rearrange(_x, 'b s d -> b d s').shape)
    print(rearrange(_x, 'b s d -> b (s d)').shape)
    print(rearrange(_x, 'b s (h dh) -> b h s dh', h=12).shape)
    print(repeat(_x, 'b s d -> b copies s d', copies=3).shape)
    print(reduce(_x, 'b s d -> b d', 'mean').shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 5

    Given `_x=(16,20,128)`, write einops operations to produce:

    A. `(20,16,128)`
    B. `(16,2560)`
    C. `(16,8,20,16)` by splitting embedding into 8 heads
    D. `(16,128)` by averaging over tokens
    E. `(16,3,20,128)` by repeating sequences three times.
    """)
    return


@app.cell
def _(rearrange, reduce, repeat, t):
    _x = t.randn(16, 20, 128)
    # YOUR WORK

    print(rearrange(_x,"a b c -> b a c").shape)
    print(rearrange(_x,"a b c -> a (b c)").shape)
    print(rearrange(_x,"a b (c d) -> a c b d", c =8).shape)
    print(reduce(_x,"a b c -> a c", "mean").shape)
    print(repeat(_x,"a b c -> a rep b c", rep = 3).shape)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Solution 5 — click to reveal": mo.md(r"""
        **Solution 5**

        ```python
        rearrange(_x, "b s d -> s b d")
        rearrange(_x, "b s d -> b (s d)")
        rearrange(_x, "b s (h dh) -> b h s dh", h=8)
        reduce(_x, "b s d -> b d", "mean")
        repeat(_x, "b s d -> b copies s d", copies=3)
        ```
        """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Final challenge: pairwise dot products (10 min)

    `_A=(10,3)` and `_B=(7,3)` are collections of 3D vectors. Compute every dot product to produce a tensor shaped `(10,7)`:

    $$C_{ij}=\sum_k A_{ik} B_{jk}$$

    Implement it **three ways**: (1) broadcasting + `.sum`, (2) broadcasting + `einops.reduce`, (3) matrix multiplication. Verify with `t.allclose`.
    """)
    return


@app.cell
def _(reduce, t):
    _A = t.randn(10, 3)
    _B = t.randn(7, 3)
    # YOUR WORK
    #(10,3)->(10,1,3)
    #(7,3)->(1,7,3)
    #(10,7,3).sum(dim=-1)
    _C_broadcast = (_A[:,None,:]*_B).sum(dim=-1)
    _C_einops = reduce((_A[:,None,:]*_B),'a b c -> a b', 'sum')
    _C_matmul = _A@_B.T
    assert _C_broadcast.shape == (10, 7)
    assert t.allclose(_C_broadcast, _C_einops)
    assert t.allclose(_C_broadcast, _C_matmul, atol=1e-5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Solution 6 — click to reveal": mo.md(r"""
        **Solution 6**

        ```python
        _products = _A[:,None,:] * _B[None,:,:]       # (10,7,3)
        _C_broadcast = _products.sum(dim=-1)         # (10,7)
        _C_einops = reduce(_products, "a b d -> a b", "sum")
        _C_matmul = _A @ _B.T
        assert t.allclose(_C_broadcast, _C_einops)
        assert t.allclose(_C_broadcast, _C_matmul, atol=1e-5)
        ```
        """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bonus · Einsum (optional)

    Repeated labels not in the output are summed over:

    ```python
    _C = t.einsum("ik,jk->ij", _A, _B)
    ```

    Try this in the final challenge's starter cell, where `_A` and `_B` are defined.

    For attention scores, `(batch, heads, query, d)` and `(batch, heads, key, d)` contract over `d` to yield `(batch, heads, query, key)`.
    """)
    return


@app.cell
def _(t):
    _Q = t.randn(2, 4, 10, 64)
    K = t.randn(2, 4, 10, 64)
    scores = t.einsum('b h q d, b h k d -> b h q k', _Q, K)
    print(scores.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cheat sheet & ARENA transfer

    | Task | PyTorch | Einops |
    |---|---|---|
    | Swap axes | `x.transpose(1,2)` | `rearrange(x, "b s d -> b d s")` |
    | Flatten | `x.reshape(b,-1)` | `rearrange(x, "b s d -> b (s d)")` |
    | Add singleton axis | `x[:,None,:]` | `rearrange(x, "b d -> b 1 d")` |
    | Add repeated axis | `x[:,None,:,:].repeat(1,4,1,1)` | `repeat(x, "b s d -> b n s d", n=4)` |
    | Mean across tokens | `x.mean(dim=1)` | `reduce(x, "b s d -> b d", "mean")` |
    | Matrix multiply | `A @ B` | — |

    **Return to ARENA:** before coding the ray–segment intersection, predict each shape:

    ```text
    Ray directions:       (nrays, 2)
    Segment directions:   (nsegments, 2)
    Pairwise matrices:    (nrays, nsegments, 2, 2)
    Right-hand sides:     (nrays, nsegments, 2)
    Solutions:            (nrays, nsegments, 2)
    Pairwise intersections:(nrays, nsegments)
    Final result:         (nrays,)
    ```

    **Self-check:** Can you explain which axis is added, broadcast, contracted, or reduced at each step?

    References: [PyTorch broadcasting](https://docs.pytorch.org/docs/stable/notes/broadcasting.html) · [Einops basics](https://einops.rocks/1-einops-basics/)
    """)
    return


if __name__ == "__main__":
    app.run()
