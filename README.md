# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py




python project/parallel_check.py
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/jacksoncamp/Desktop/CornellTech/CS 5781/mod3-jacksoncamp42/minitorch/fast_ops.py (163) 
----------------------------------------------------------------------------------|loop #ID
    def _map(                                                                     | 
        out: Storage,                                                             | 
        out_shape: Shape,                                                         | 
        out_strides: Strides,                                                     | 
        in_storage: Storage,                                                      | 
        in_shape: Shape,                                                          | 
        in_strides: Strides,                                                      | 
    ) -> None:                                                                    | 
        # TODO: Implement for Task 3.1.                                           | 
        # Check if shape lengths match first                                      | 
        if len(out_shape) != len(in_shape):                                       | 
            # Non-aligned case                                                    | 
            for i in prange(len(out)):--------------------------------------------| #6
                out_index = np.zeros(MAX_DIMS, np.int32)--------------------------| #0
                in_index = np.zeros(MAX_DIMS, np.int32)---------------------------| #1
                to_index(i, out_shape, out_index)                                 | 
                broadcast_index(out_index, out_shape, in_shape, in_index)         | 
                o = index_to_position(out_index, out_strides)                     | 
                j = index_to_position(in_index, in_strides)                       | 
                out[o] = fn(in_storage[j])                                        | 
            return                                                                | 
                                                                                  | 
        # Check strides and shapes are aligned                                    | 
        aligned = True                                                            | 
        for i in range(len(out_shape)):                                           | 
            if out_shape[i] != in_shape[i] or out_strides[i] != in_strides[i]:    | 
                aligned = False                                                   | 
                break                                                             | 
                                                                                  | 
        if aligned:                                                               | 
            # Fast path - direct indexing                                         | 
            for i in prange(len(out)):--------------------------------------------| #4
                out[i] = fn(in_storage[i])                                        | 
        else:                                                                     | 
            # Regular path with explicit indexing                                 | 
            for i in prange(len(out)):--------------------------------------------| #5
                out_index = np.zeros(MAX_DIMS, np.int32)--------------------------| #2
                in_index = np.zeros(MAX_DIMS, np.int32)---------------------------| #3
                to_index(i, out_shape, out_index)                                 | 
                broadcast_index(out_index, out_shape, in_shape, in_index)         | 
                o = index_to_position(out_index, out_strides)                     | 
                j = index_to_position(in_index, in_strides)                       | 
                out[o] = fn(in_storage[j])                                        | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
+--2 has the following loops fused into it:
   +--3 (fused)
Following the attempted fusion of parallel for-loops there are 5 parallel for-
loop(s) (originating from loops labelled: #6, #0, #4, #5, #2).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--5 is a parallel loop
   +--2 --> rewritten as a serial loop
+--6 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--6 (parallel)
   +--0 (parallel)
   +--1 (parallel)


Parallel region 1:
+--5 (parallel)
   +--2 (parallel)
   +--3 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--6 (parallel)
   +--0 (serial, fused with loop(s): 1)


Parallel region 1:
+--5 (parallel)
   +--2 (serial, fused with loop(s): 3)


 
Parallel region 0 (loop #6) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#6).
 
Parallel region 1 (loop #5) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#5).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (176) is hoisted out of the 
parallel loop labelled #6 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (177) is hoisted out of the 
parallel loop labelled #6 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: in_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (199) is hoisted out of the 
parallel loop labelled #5 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (200) is hoisted out of the 
parallel loop labelled #5 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: in_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (233)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/jacksoncamp/Desktop/CornellTech/CS 5781/mod3-jacksoncamp42/minitorch/fast_ops.py (233) 
--------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                   | 
        out: Storage,                                                           | 
        out_shape: Shape,                                                       | 
        out_strides: Strides,                                                   | 
        a_storage: Storage,                                                     | 
        a_shape: Shape,                                                         | 
        a_strides: Strides,                                                     | 
        b_storage: Storage,                                                     | 
        b_shape: Shape,                                                         | 
        b_strides: Strides,                                                     | 
    ) -> None:                                                                  | 
        # TODO: Implement for Task 3.1.                                         | 
        # Check if shape lengths match                                          | 
        if len(out_shape) != len(a_shape) or len(out_shape) != len(b_shape):    | 
            # Non-aligned case                                                  | 
            for i in prange(len(out)):------------------------------------------| #15
                out_index = np.zeros(MAX_DIMS, np.int32)------------------------| #7
                a_index = np.zeros(MAX_DIMS, np.int32)--------------------------| #8
                b_index = np.zeros(MAX_DIMS, np.int32)--------------------------| #9
                to_index(i, out_shape, out_index)                               | 
                broadcast_index(out_index, out_shape, a_shape, a_index)         | 
                broadcast_index(out_index, out_shape, b_shape, b_index)         | 
                o = index_to_position(out_index, out_strides)                   | 
                a = index_to_position(a_index, a_strides)                       | 
                b = index_to_position(b_index, b_strides)                       | 
                out[o] = fn(a_storage[a], b_storage[b])                         | 
            return                                                              | 
                                                                                | 
        # Check strides and shapes are aligned                                  | 
        aligned = True                                                          | 
        for i in range(len(out_shape)):                                         | 
            if (                                                                | 
                out_shape[i] != a_shape[i]                                      | 
                or out_shape[i] != b_shape[i]                                   | 
                or out_strides[i] != a_strides[i]                               | 
                or out_strides[i] != b_strides[i]                               | 
            ):                                                                  | 
                aligned = False                                                 | 
                break                                                           | 
                                                                                | 
        if aligned:                                                             | 
            # Fast path - direct indexing                                       | 
            for i in prange(len(out)):------------------------------------------| #13
                out[i] = fn(a_storage[i], b_storage[i])                         | 
        else:                                                                   | 
            # Regular path with explicit indexing                               | 
            for i in prange(len(out)):------------------------------------------| #14
                out_index = np.zeros(MAX_DIMS, np.int32)------------------------| #10
                a_index = np.zeros(MAX_DIMS, np.int32)--------------------------| #11
                b_index = np.zeros(MAX_DIMS, np.int32)--------------------------| #12
                to_index(i, out_shape, out_index)                               | 
                broadcast_index(out_index, out_shape, a_shape, a_index)         | 
                broadcast_index(out_index, out_shape, b_shape, b_index)         | 
                o = index_to_position(out_index, out_strides)                   | 
                a = index_to_position(a_index, a_strides)                       | 
                b = index_to_position(b_index, b_strides)                       | 
                out[o] = fn(a_storage[a], b_storage[b])                         | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--10 has the following loops fused into it:
   +--11 (fused)
   +--12 (fused)
+--7 has the following loops fused into it:
   +--8 (fused)
   +--9 (fused)
Following the attempted fusion of parallel for-loops there are 5 parallel for-
loop(s) (originating from loops labelled: #15, #7, #13, #14, #10).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--14 is a parallel loop
   +--10 --> rewritten as a serial loop
+--15 is a parallel loop
   +--7 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--15 (parallel)
   +--7 (parallel)
   +--8 (parallel)
   +--9 (parallel)


Parallel region 1:
+--14 (parallel)
   +--10 (parallel)
   +--11 (parallel)
   +--12 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--15 (parallel)
   +--7 (serial, fused with loop(s): 8, 9)


Parallel region 1:
+--14 (parallel)
   +--10 (serial, fused with loop(s): 11, 12)


 
Parallel region 0 (loop #15) had 2 loop(s) fused and 1 loop(s) serialized as 
part of the larger parallel loop (#15).
 
Parallel region 1 (loop #14) had 2 loop(s) fused and 1 loop(s) serialized as 
part of the larger parallel loop (#14).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (249) is hoisted out of the 
parallel loop labelled #15 (it will be performed before the loop is executed and
 reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (250) is hoisted out of the 
parallel loop labelled #15 (it will be performed before the loop is executed and
 reused inside the loop):
   Allocation:: a_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (251) is hoisted out of the 
parallel loop labelled #15 (it will be performed before the loop is executed and
 reused inside the loop):
   Allocation:: b_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (280) is hoisted out of the 
parallel loop labelled #14 (it will be performed before the loop is executed and
 reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (281) is hoisted out of the 
parallel loop labelled #14 (it will be performed before the loop is executed and
 reused inside the loop):
   Allocation:: a_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (282) is hoisted out of the 
parallel loop labelled #14 (it will be performed before the loop is executed and
 reused inside the loop):
   Allocation:: b_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (315)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/jacksoncamp/Desktop/CornellTech/CS 5781/mod3-jacksoncamp42/minitorch/fast_ops.py (315) 
-----------------------------------------------------------------|loop #ID
    def _reduce(                                                 | 
        out: Storage,                                            | 
        out_shape: Shape,                                        | 
        out_strides: Strides,                                    | 
        a_storage: Storage,                                      | 
        a_shape: Shape,                                          | 
        a_strides: Strides,                                      | 
        reduce_dim: int,                                         | 
    ) -> None:                                                   | 
        # TODO: Implement for Task 3.1.                          | 
        # Use outer loop parallel, inner reduction sequential    | 
        for i in prange(len(out)):-------------------------------| #17
            out_index = np.zeros(MAX_DIMS, np.int32)-------------| #16
            to_index(i, out_shape, out_index)                    | 
            o = index_to_position(out_index, out_strides)        | 
                                                                 | 
            # Sequential reduction for numerical stability       | 
            for j in range(a_shape[reduce_dim]):                 | 
                out_index[reduce_dim] = j                        | 
                k = index_to_position(out_index, a_strides)      | 
                out[o] = fn(out[o], a_storage[k])                | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #17, #16).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--17 is a parallel loop
   +--16 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--17 (parallel)
   +--16 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--17 (parallel)
   +--16 (serial)


 
Parallel region 0 (loop #17) had 0 loop(s) fused and 1 loop(s) serialized as 
part of the larger parallel loop (#17).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (327) is hoisted out of the 
parallel loop labelled #17 (it will be performed before the loop is executed and
 reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (340)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/jacksoncamp/Desktop/CornellTech/CS 5781/mod3-jacksoncamp42/minitorch/fast_ops.py (340) 
----------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                | 
    out: Storage,                                                           | 
    out_shape: Shape,                                                       | 
    out_strides: Strides,                                                   | 
    a_storage: Storage,                                                     | 
    a_shape: Shape,                                                         | 
    a_strides: Strides,                                                     | 
    b_storage: Storage,                                                     | 
    b_shape: Shape,                                                         | 
    b_strides: Strides,                                                     | 
) -> None:                                                                  | 
    """NUMBA tensor matrix multiply function.                               | 
                                                                            | 
    Should work for any tensor shapes that broadcast as long as             | 
                                                                            | 
    ```                                                                     | 
    assert a_shape[-1] == b_shape[-2]                                       | 
    ```                                                                     | 
                                                                            | 
    Optimizations:                                                          | 
                                                                            | 
    * Outer loop in parallel                                                | 
    * No index buffers or function calls                                    | 
    * Inner loop should have no global writes, 1 multiply.                  | 
                                                                            | 
                                                                            | 
    Args:                                                                   | 
    ----                                                                    | 
        out (Storage): storage for `out` tensor                             | 
        out_shape (Shape): shape for `out` tensor                           | 
        out_strides (Strides): strides for `out` tensor                     | 
        a_storage (Storage): storage for `a` tensor                         | 
        a_shape (Shape): shape for `a` tensor                               | 
        a_strides (Strides): strides for `a` tensor                         | 
        b_storage (Storage): storage for `b` tensor                         | 
        b_shape (Shape): shape for `b` tensor                               | 
        b_strides (Strides): strides for `b` tensor                         | 
                                                                            | 
    Returns:                                                                | 
    -------                                                                 | 
        None : Fills in `out`                                               | 
                                                                            | 
    """                                                                     | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                  | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                  | 
                                                                            | 
    # TODO: Implement for Task 3.2.                                         | 
    # Extract dimensions                                                    | 
    batch_size = out_shape[0]                                               | 
    M = out_shape[1]                                                        | 
    N = out_shape[2]                                                        | 
    K = a_shape[2]  # a_shape[-1] == b_shape[-2]                            | 
                                                                            | 
    for b in prange(batch_size):--------------------------------------------| #18
        # Calculate base strides for current batch                          | 
        a_batch_stride = b * a_strides[0] if a_shape[0] > 1 else 0          | 
        b_batch_stride = b * b_strides[0] if b_shape[0] > 1 else 0          | 
        out_batch_stride = b * out_strides[0] if out_shape[0] > 1 else 0    | 
                                                                            | 
        for m in range(M):                                                  | 
            # Calculate base strides for current row                        | 
            a_row_stride = a_batch_stride + m * a_strides[1]                | 
            out_row_stride = out_batch_stride + m * out_strides[1]          | 
                                                                            | 
            for n in range(N):                                              | 
                # Calculate base strides for current column                 | 
                b_col_stride = b_batch_stride + n * b_strides[2]            | 
                                                                            | 
                # Initialize the accumulator                                | 
                acc = 0.0                                                   | 
                                                                            | 
                # Perform the dot product                                   | 
                for k in range(K):                                          | 
                    a_idx = a_row_stride + k * a_strides[2]                 | 
                    b_idx = b_col_stride + k * b_strides[1]                 | 
                    acc += a_storage[a_idx] * b_storage[b_idx]              | 
                                                                            | 
                # Assign the computed value to the output tensor            | 
                out_idx = out_row_stride + n * out_strides[2]               | 
                out[out_idx] = acc                                          | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #18).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None


# !python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
Epoch   0 | Loss       6.20 | Correct   47 | Time/epoch 0.000s
Epoch  10 | Loss       1.40 | Correct   48 | Time/epoch 1.977s
Epoch  20 | Loss       2.06 | Correct   48 | Time/epoch 2.015s
Epoch  30 | Loss       1.07 | Correct   49 | Time/epoch 2.045s
Epoch  40 | Loss       0.42 | Correct   49 | Time/epoch 2.023s
Epoch  50 | Loss       1.17 | Correct   49 | Time/epoch 2.025s
Epoch  60 | Loss       0.86 | Correct   49 | Time/epoch 2.015s
Epoch  70 | Loss       0.32 | Correct   49 | Time/epoch 2.012s
Epoch  80 | Loss       0.05 | Correct   49 | Time/epoch 2.013s
Epoch  90 | Loss       1.36 | Correct   49 | Time/epoch 2.004s
Epoch 100 | Loss       0.63 | Correct   49 | Time/epoch 2.006s
Epoch 110 | Loss       0.87 | Correct   50 | Time/epoch 2.008s
Epoch 120 | Loss       0.10 | Correct   49 | Time/epoch 2.003s
Epoch 130 | Loss       0.28 | Correct   50 | Time/epoch 2.003s
Epoch 140 | Loss       1.45 | Correct   50 | Time/epoch 2.000s
Epoch 150 | Loss       0.74 | Correct   49 | Time/epoch 2.000s
Epoch 160 | Loss       0.01 | Correct   47 | Time/epoch 2.000s
Epoch 170 | Loss       0.09 | Correct   50 | Time/epoch 1.996s
Epoch 180 | Loss       0.44 | Correct   49 | Time/epoch 2.000s
Epoch 190 | Loss       0.69 | Correct   50 | Time/epoch 2.002s
Epoch 200 | Loss       0.12 | Correct   50 | Time/epoch 2.000s
Epoch 210 | Loss       0.05 | Correct   50 | Time/epoch 2.002s
Epoch 220 | Loss       0.11 | Correct   50 | Time/epoch 2.004s
Epoch 230 | Loss       1.04 | Correct   50 | Time/epoch 2.002s
Epoch 240 | Loss       0.38 | Correct   49 | Time/epoch 2.003s
Epoch 250 | Loss       0.76 | Correct   50 | Time/epoch 2.003s
Epoch 260 | Loss       1.02 | Correct   49 | Time/epoch 2.002s
Epoch 270 | Loss       0.06 | Correct   50 | Time/epoch 2.002s
Epoch 280 | Loss       0.00 | Correct   50 | Time/epoch 2.000s
Epoch 290 | Loss       0.10 | Correct   50 | Time/epoch 2.000s
Epoch 300 | Loss       0.73 | Correct   50 | Time/epoch 2.001s
Epoch 310 | Loss       1.75 | Correct   49 | Time/epoch 1.999s
Epoch 320 | Loss       0.14 | Correct   50 | Time/epoch 1.999s
Epoch 330 | Loss       0.12 | Correct   50 | Time/epoch 1.998s
Epoch 340 | Loss       0.16 | Correct   49 | Time/epoch 1.998s
Epoch 350 | Loss       0.21 | Correct   50 | Time/epoch 1.999s
Epoch 360 | Loss       1.76 | Correct   50 | Time/epoch 1.997s
Epoch 370 | Loss       0.42 | Correct   50 | Time/epoch 1.998s
Epoch 380 | Loss       0.28 | Correct   50 | Time/epoch 1.999s
Epoch 390 | Loss       0.29 | Correct   50 | Time/epoch 1.997s
Epoch 400 | Loss       0.01 | Correct   50 | Time/epoch 1.998s
Epoch 410 | Loss       1.28 | Correct   49 | Time/epoch 1.997s
Epoch 420 | Loss       0.01 | Correct   49 | Time/epoch 1.997s
Epoch 430 | Loss       1.47 | Correct   49 | Time/epoch 1.997s
Epoch 440 | Loss       0.14 | Correct   50 | Time/epoch 1.996s
Epoch 450 | Loss       0.70 | Correct   49 | Time/epoch 1.996s
Epoch 460 | Loss       0.12 | Correct   49 | Time/epoch 1.997s
Epoch 470 | Loss       0.74 | Correct   49 | Time/epoch 1.995s
Epoch 480 | Loss       0.04 | Correct   50 | Time/epoch 1.996s
Epoch 490 | Loss       0.78 | Correct   50 | Time/epoch 1.996s

# !python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
Epoch   0 | Loss       3.92 | Correct   42 | Time/epoch 0.000s
Epoch  10 | Loss       0.63 | Correct   49 | Time/epoch 0.140s
Epoch  20 | Loss       0.86 | Correct   48 | Time/epoch 0.140s
Epoch  30 | Loss       1.13 | Correct   49 | Time/epoch 0.140s
Epoch  40 | Loss       0.71 | Correct   47 | Time/epoch 0.139s
Epoch  50 | Loss       1.31 | Correct   49 | Time/epoch 0.143s
Epoch  60 | Loss       1.64 | Correct   48 | Time/epoch 0.158s
Epoch  70 | Loss       0.63 | Correct   49 | Time/epoch 0.155s
Epoch  80 | Loss       1.25 | Correct   48 | Time/epoch 0.153s
Epoch  90 | Loss       1.26 | Correct   50 | Time/epoch 0.151s
Epoch 100 | Loss       0.74 | Correct   49 | Time/epoch 0.150s
Epoch 110 | Loss       0.84 | Correct   50 | Time/epoch 0.150s
Epoch 120 | Loss       0.22 | Correct   50 | Time/epoch 0.149s
Epoch 130 | Loss       0.82 | Correct   49 | Time/epoch 0.149s
Epoch 140 | Loss       1.10 | Correct   50 | Time/epoch 0.155s
Epoch 150 | Loss       1.64 | Correct   49 | Time/epoch 0.154s
Epoch 160 | Loss       0.23 | Correct   49 | Time/epoch 0.153s
Epoch 170 | Loss       0.22 | Correct   50 | Time/epoch 0.152s
Epoch 180 | Loss       0.64 | Correct   50 | Time/epoch 0.151s
Epoch 190 | Loss       0.04 | Correct   50 | Time/epoch 0.151s
Epoch 200 | Loss       0.83 | Correct   48 | Time/epoch 0.150s
Epoch 210 | Loss       0.36 | Correct   49 | Time/epoch 0.149s
Epoch 220 | Loss       0.44 | Correct   50 | Time/epoch 0.153s
Epoch 230 | Loss       1.10 | Correct   49 | Time/epoch 0.153s
Epoch 240 | Loss       0.33 | Correct   49 | Time/epoch 0.153s
Epoch 250 | Loss       0.90 | Correct   49 | Time/epoch 0.152s
Epoch 260 | Loss       0.02 | Correct   50 | Time/epoch 0.152s
Epoch 270 | Loss       0.10 | Correct   50 | Time/epoch 0.151s
Epoch 280 | Loss       0.11 | Correct   50 | Time/epoch 0.151s
Epoch 290 | Loss       0.20 | Correct   50 | Time/epoch 0.150s
Epoch 300 | Loss       0.37 | Correct   49 | Time/epoch 0.153s
Epoch 310 | Loss       0.12 | Correct   50 | Time/epoch 0.153s
Epoch 320 | Loss       0.38 | Correct   49 | Time/epoch 0.153s
Epoch 330 | Loss       1.15 | Correct   50 | Time/epoch 0.152s
Epoch 340 | Loss       0.80 | Correct   49 | Time/epoch 0.152s
Epoch 350 | Loss       1.14 | Correct   49 | Time/epoch 0.152s
Epoch 360 | Loss       0.08 | Correct   50 | Time/epoch 0.151s
Epoch 370 | Loss       0.01 | Correct   50 | Time/epoch 0.151s
Epoch 380 | Loss       0.03 | Correct   50 | Time/epoch 0.152s
Epoch 390 | Loss       0.68 | Correct   50 | Time/epoch 0.153s
Epoch 400 | Loss       0.01 | Correct   49 | Time/epoch 0.153s
Epoch 410 | Loss       0.85 | Correct   49 | Time/epoch 0.152s
Epoch 420 | Loss       0.90 | Correct   49 | Time/epoch 0.152s
Epoch 430 | Loss       0.12 | Correct   50 | Time/epoch 0.152s
Epoch 440 | Loss       0.96 | Correct   49 | Time/epoch 0.151s
Epoch 450 | Loss       0.68 | Correct   49 | Time/epoch 0.151s
Epoch 460 | Loss       0.76 | Correct   49 | Time/epoch 0.152s
Epoch 470 | Loss       0.83 | Correct   49 | Time/epoch 0.153s
Epoch 480 | Loss       1.00 | Correct   50 | Time/epoch 0.152s
Epoch 490 | Loss       1.18 | Correct   49 | Time/epoch 0.152s

# !python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
Epoch   0 | Loss       7.77 | Correct   40 | Time/epoch 0.000s
Epoch  10 | Loss       4.37 | Correct   42 | Time/epoch 1.963s
Epoch  20 | Loss       5.29 | Correct   45 | Time/epoch 2.011s
Epoch  30 | Loss       3.33 | Correct   45 | Time/epoch 2.018s
Epoch  40 | Loss       2.57 | Correct   48 | Time/epoch 2.006s
Epoch  50 | Loss       2.18 | Correct   47 | Time/epoch 2.015s
Epoch  60 | Loss       2.18 | Correct   49 | Time/epoch 2.014s
Epoch  70 | Loss       2.84 | Correct   50 | Time/epoch 2.013s
Epoch  80 | Loss       2.65 | Correct   49 | Time/epoch 2.019s
Epoch  90 | Loss       1.53 | Correct   50 | Time/epoch 2.018s
Epoch 100 | Loss       1.31 | Correct   49 | Time/epoch 2.018s
Epoch 110 | Loss       1.47 | Correct   47 | Time/epoch 2.020s
Epoch 120 | Loss       2.02 | Correct   48 | Time/epoch 2.015s
Epoch 130 | Loss       1.79 | Correct   50 | Time/epoch 2.018s
Epoch 140 | Loss       0.91 | Correct   49 | Time/epoch 2.027s
Epoch 150 | Loss       0.24 | Correct   50 | Time/epoch 2.034s
Epoch 160 | Loss       0.47 | Correct   50 | Time/epoch 2.038s
Epoch 170 | Loss       1.37 | Correct   50 | Time/epoch 2.043s
Epoch 180 | Loss       2.98 | Correct   46 | Time/epoch 2.049s
Epoch 190 | Loss       0.87 | Correct   50 | Time/epoch 2.051s
Epoch 200 | Loss       0.58 | Correct   49 | Time/epoch 2.053s
Epoch 210 | Loss       0.51 | Correct   50 | Time/epoch 2.054s
Epoch 220 | Loss       1.27 | Correct   49 | Time/epoch 2.051s
Epoch 230 | Loss       0.66 | Correct   50 | Time/epoch 2.050s
Epoch 240 | Loss       1.56 | Correct   50 | Time/epoch 2.051s
Epoch 250 | Loss       0.64 | Correct   50 | Time/epoch 2.049s
Epoch 260 | Loss       0.57 | Correct   50 | Time/epoch 2.048s
Epoch 270 | Loss       0.71 | Correct   50 | Time/epoch 2.050s
Epoch 280 | Loss       0.07 | Correct   50 | Time/epoch 2.048s
Epoch 290 | Loss       1.10 | Correct   50 | Time/epoch 2.050s
Epoch 300 | Loss       0.22 | Correct   50 | Time/epoch 2.052s
Epoch 310 | Loss       0.31 | Correct   50 | Time/epoch 2.050s
Epoch 320 | Loss       0.26 | Correct   50 | Time/epoch 2.050s
Epoch 330 | Loss       0.87 | Correct   50 | Time/epoch 2.051s
Epoch 340 | Loss       0.06 | Correct   50 | Time/epoch 2.051s
Epoch 350 | Loss       0.86 | Correct   50 | Time/epoch 2.052s
Epoch 360 | Loss       0.17 | Correct   50 | Time/epoch 2.053s
Epoch 370 | Loss       0.13 | Correct   50 | Time/epoch 2.056s
Epoch 380 | Loss       0.41 | Correct   50 | Time/epoch 2.055s
Epoch 390 | Loss       0.23 | Correct   50 | Time/epoch 2.055s
Epoch 400 | Loss       0.11 | Correct   50 | Time/epoch 2.058s
Epoch 410 | Loss       0.32 | Correct   50 | Time/epoch 2.061s
Epoch 420 | Loss       0.86 | Correct   50 | Time/epoch 2.065s
Epoch 430 | Loss       0.11 | Correct   50 | Time/epoch 2.070s
Epoch 440 | Loss       0.22 | Correct   50 | Time/epoch 2.073s
Epoch 450 | Loss       0.33 | Correct   50 | Time/epoch 2.077s
Epoch 460 | Loss       0.01 | Correct   50 | Time/epoch 2.078s
Epoch 470 | Loss       0.19 | Correct   50 | Time/epoch 2.080s
Epoch 480 | Loss       0.13 | Correct   50 | Time/epoch 2.082s
Epoch 490 | Loss       0.20 | Correct   50 | Time/epoch 2.083s

# !python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
Epoch   0 | Loss       6.41 | Correct   32 | Time/epoch 0.000s
Epoch  10 | Loss       4.87 | Correct   40 | Time/epoch 0.150s
Epoch  20 | Loss       5.71 | Correct   45 | Time/epoch 0.152s
Epoch  30 | Loss       2.89 | Correct   44 | Time/epoch 0.160s
Epoch  40 | Loss       2.85 | Correct   46 | Time/epoch 0.183s
Epoch  50 | Loss       2.51 | Correct   45 | Time/epoch 0.176s
Epoch  60 | Loss       1.70 | Correct   48 | Time/epoch 0.172s
Epoch  70 | Loss       3.23 | Correct   47 | Time/epoch 0.169s
Epoch  80 | Loss       1.90 | Correct   48 | Time/epoch 0.167s
Epoch  90 | Loss       3.14 | Correct   49 | Time/epoch 0.165s
Epoch 100 | Loss       1.04 | Correct   50 | Time/epoch 0.164s
Epoch 110 | Loss       1.48 | Correct   49 | Time/epoch 0.169s
Epoch 120 | Loss       1.61 | Correct   49 | Time/epoch 0.173s
Epoch 130 | Loss       0.87 | Correct   49 | Time/epoch 0.172s
Epoch 140 | Loss       0.93 | Correct   50 | Time/epoch 0.170s
Epoch 150 | Loss       0.66 | Correct   49 | Time/epoch 0.169s
Epoch 160 | Loss       0.36 | Correct   50 | Time/epoch 0.168s
Epoch 170 | Loss       0.56 | Correct   50 | Time/epoch 0.167s
Epoch 180 | Loss       1.05 | Correct   49 | Time/epoch 0.167s
Epoch 190 | Loss       1.01 | Correct   50 | Time/epoch 0.173s
Epoch 200 | Loss       0.61 | Correct   50 | Time/epoch 0.172s
Epoch 210 | Loss       0.44 | Correct   50 | Time/epoch 0.171s
Epoch 220 | Loss       0.55 | Correct   50 | Time/epoch 0.170s
Epoch 230 | Loss       0.90 | Correct   50 | Time/epoch 0.169s
Epoch 240 | Loss       0.13 | Correct   50 | Time/epoch 0.169s
Epoch 250 | Loss       0.85 | Correct   50 | Time/epoch 0.168s
Epoch 260 | Loss       0.55 | Correct   50 | Time/epoch 0.171s
Epoch 270 | Loss       0.67 | Correct   49 | Time/epoch 0.171s
Epoch 280 | Loss       0.51 | Correct   50 | Time/epoch 0.170s
Epoch 290 | Loss       0.41 | Correct   50 | Time/epoch 0.170s
Epoch 300 | Loss       0.74 | Correct   50 | Time/epoch 0.169s
Epoch 310 | Loss       0.15 | Correct   50 | Time/epoch 0.169s
Epoch 320 | Loss       0.04 | Correct   50 | Time/epoch 0.168s
Epoch 330 | Loss       0.06 | Correct   50 | Time/epoch 0.167s
Epoch 340 | Loss       1.07 | Correct   50 | Time/epoch 0.170s
Epoch 350 | Loss       0.96 | Correct   50 | Time/epoch 0.170s
Epoch 360 | Loss       0.66 | Correct   50 | Time/epoch 0.169s
Epoch 370 | Loss       0.11 | Correct   50 | Time/epoch 0.169s
Epoch 380 | Loss       0.29 | Correct   50 | Time/epoch 0.168s
Epoch 390 | Loss       0.59 | Correct   50 | Time/epoch 0.168s
Epoch 400 | Loss       0.27 | Correct   50 | Time/epoch 0.167s
Epoch 410 | Loss       0.17 | Correct   50 | Time/epoch 0.168s
Epoch 420 | Loss       0.95 | Correct   50 | Time/epoch 0.169s
Epoch 430 | Loss       0.16 | Correct   50 | Time/epoch 0.169s
Epoch 440 | Loss       0.09 | Correct   50 | Time/epoch 0.168s
Epoch 450 | Loss       0.72 | Correct   50 | Time/epoch 0.168s
Epoch 460 | Loss       0.31 | Correct   50 | Time/epoch 0.168s
Epoch 470 | Loss       0.23 | Correct   50 | Time/epoch 0.167s
Epoch 480 | Loss       0.13 | Correct   50 | Time/epoch 0.167s
Epoch 490 | Loss       0.51 | Correct   50 | Time/epoch 0.169s

# !python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
Epoch   0 | Loss       5.91 | Correct   32 | Time/epoch 0.000s
Epoch  10 | Loss       5.88 | Correct   40 | Time/epoch 2.228s
Epoch  20 | Loss       5.38 | Correct   46 | Time/epoch 2.222s
Epoch  30 | Loss       3.45 | Correct   45 | Time/epoch 2.201s
Epoch  40 | Loss       2.47 | Correct   48 | Time/epoch 2.180s
Epoch  50 | Loss       2.32 | Correct   47 | Time/epoch 2.178s
Epoch  60 | Loss       1.69 | Correct   48 | Time/epoch 2.177s
Epoch  70 | Loss       2.40 | Correct   49 | Time/epoch 2.168s
Epoch  80 | Loss       1.89 | Correct   48 | Time/epoch 2.168s
Epoch  90 | Loss       2.83 | Correct   49 | Time/epoch 2.168s
Epoch 100 | Loss       1.69 | Correct   48 | Time/epoch 2.169s
Epoch 110 | Loss       2.29 | Correct   48 | Time/epoch 2.161s
Epoch 120 | Loss       0.64 | Correct   48 | Time/epoch 2.159s
Epoch 130 | Loss       1.91 | Correct   49 | Time/epoch 2.160s
Epoch 140 | Loss       1.32 | Correct   49 | Time/epoch 2.161s
Epoch 150 | Loss       2.47 | Correct   50 | Time/epoch 2.159s
Epoch 160 | Loss       1.08 | Correct   49 | Time/epoch 2.155s
Epoch 170 | Loss       0.58 | Correct   48 | Time/epoch 2.154s
Epoch 180 | Loss       0.90 | Correct   48 | Time/epoch 2.154s
Epoch 190 | Loss       0.40 | Correct   50 | Time/epoch 2.151s
Epoch 200 | Loss       1.63 | Correct   50 | Time/epoch 2.146s
Epoch 210 | Loss       0.42 | Correct   50 | Time/epoch 2.147s
Epoch 220 | Loss       0.54 | Correct   50 | Time/epoch 2.146s
Epoch 230 | Loss       0.99 | Correct   50 | Time/epoch 2.142s
Epoch 240 | Loss       0.37 | Correct   50 | Time/epoch 2.141s
Epoch 250 | Loss       1.21 | Correct   50 | Time/epoch 2.141s
Epoch 260 | Loss       0.31 | Correct   48 | Time/epoch 2.141s
Epoch 270 | Loss       0.45 | Correct   50 | Time/epoch 2.140s
Epoch 280 | Loss       0.16 | Correct   50 | Time/epoch 2.140s
Epoch 290 | Loss       1.03 | Correct   49 | Time/epoch 2.142s
Epoch 300 | Loss       0.50 | Correct   49 | Time/epoch 2.143s
Epoch 310 | Loss       0.83 | Correct   50 | Time/epoch 2.145s
Epoch 320 | Loss       0.29 | Correct   50 | Time/epoch 2.144s
Epoch 330 | Loss       0.81 | Correct   50 | Time/epoch 2.145s
Epoch 340 | Loss       0.52 | Correct   50 | Time/epoch 2.145s
Epoch 350 | Loss       0.58 | Correct   50 | Time/epoch 2.146s
Epoch 360 | Loss       0.88 | Correct   50 | Time/epoch 2.144s
Epoch 370 | Loss       0.46 | Correct   50 | Time/epoch 2.144s
Epoch 380 | Loss       0.44 | Correct   50 | Time/epoch 2.144s
Epoch 390 | Loss       0.16 | Correct   50 | Time/epoch 2.145s
Epoch 400 | Loss       0.88 | Correct   50 | Time/epoch 2.144s
Epoch 410 | Loss       0.08 | Correct   50 | Time/epoch 2.144s
Epoch 420 | Loss       0.76 | Correct   50 | Time/epoch 2.144s
Epoch 430 | Loss       0.05 | Correct   50 | Time/epoch 2.145s
Epoch 440 | Loss       0.48 | Correct   50 | Time/epoch 2.145s
Epoch 450 | Loss       0.78 | Correct   50 | Time/epoch 2.144s
Epoch 460 | Loss       0.67 | Correct   50 | Time/epoch 2.145s
Epoch 470 | Loss       0.82 | Correct   50 | Time/epoch 2.144s
Epoch 480 | Loss       0.52 | Correct   50 | Time/epoch 2.144s
Epoch 490 | Loss       0.34 | Correct   50 | Time/epoch 2.143s

# !python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
Epoch   0 | Loss       6.83 | Correct   29 | Time/epoch 0.000s
Epoch  10 | Loss       5.78 | Correct   39 | Time/epoch 0.141s
Epoch  20 | Loss       4.84 | Correct   42 | Time/epoch 0.141s
Epoch  30 | Loss       3.26 | Correct   44 | Time/epoch 0.142s
Epoch  40 | Loss       3.47 | Correct   43 | Time/epoch 0.141s
Epoch  50 | Loss       6.47 | Correct   41 | Time/epoch 0.142s
Epoch  60 | Loss       4.09 | Correct   44 | Time/epoch 0.161s
Epoch  70 | Loss       1.75 | Correct   45 | Time/epoch 0.160s
Epoch  80 | Loss       3.84 | Correct   47 | Time/epoch 0.158s
Epoch  90 | Loss       2.68 | Correct   46 | Time/epoch 0.157s
Epoch 100 | Loss       2.52 | Correct   48 | Time/epoch 0.157s
Epoch 110 | Loss       4.02 | Correct   47 | Time/epoch 0.156s
Epoch 120 | Loss       1.59 | Correct   48 | Time/epoch 0.155s
Epoch 130 | Loss       0.97 | Correct   48 | Time/epoch 0.158s
Epoch 140 | Loss       1.87 | Correct   49 | Time/epoch 0.162s
Epoch 150 | Loss       1.26 | Correct   48 | Time/epoch 0.162s
Epoch 160 | Loss       2.35 | Correct   49 | Time/epoch 0.161s
Epoch 170 | Loss       1.59 | Correct   48 | Time/epoch 0.160s
Epoch 180 | Loss       1.35 | Correct   49 | Time/epoch 0.159s
Epoch 190 | Loss       1.20 | Correct   49 | Time/epoch 0.159s
Epoch 200 | Loss       1.13 | Correct   49 | Time/epoch 0.158s
Epoch 210 | Loss       0.43 | Correct   48 | Time/epoch 0.162s
Epoch 220 | Loss       1.00 | Correct   48 | Time/epoch 0.162s
Epoch 230 | Loss       0.34 | Correct   48 | Time/epoch 0.162s
Epoch 240 | Loss       0.93 | Correct   49 | Time/epoch 0.161s
Epoch 250 | Loss       0.69 | Correct   49 | Time/epoch 0.160s
Epoch 260 | Loss       1.26 | Correct   49 | Time/epoch 0.160s
Epoch 270 | Loss       0.43 | Correct   49 | Time/epoch 0.159s
Epoch 280 | Loss       0.27 | Correct   49 | Time/epoch 0.160s
Epoch 290 | Loss       1.74 | Correct   49 | Time/epoch 0.163s
Epoch 300 | Loss       1.13 | Correct   49 | Time/epoch 0.162s
Epoch 310 | Loss       0.87 | Correct   49 | Time/epoch 0.162s
Epoch 320 | Loss       0.17 | Correct   49 | Time/epoch 0.161s
Epoch 330 | Loss       1.10 | Correct   49 | Time/epoch 0.161s
Epoch 340 | Loss       1.62 | Correct   48 | Time/epoch 0.161s
Epoch 350 | Loss       1.04 | Correct   49 | Time/epoch 0.160s
Epoch 360 | Loss       0.42 | Correct   49 | Time/epoch 0.162s
Epoch 370 | Loss       0.58 | Correct   50 | Time/epoch 0.163s
Epoch 380 | Loss       1.55 | Correct   50 | Time/epoch 0.162s
Epoch 390 | Loss       0.28 | Correct   50 | Time/epoch 0.162s
Epoch 400 | Loss       1.47 | Correct   50 | Time/epoch 0.162s
Epoch 410 | Loss       0.30 | Correct   50 | Time/epoch 0.161s
Epoch 420 | Loss       0.31 | Correct   49 | Time/epoch 0.161s
Epoch 430 | Loss       0.22 | Correct   50 | Time/epoch 0.161s
Epoch 440 | Loss       0.16 | Correct   49 | Time/epoch 0.163s
Epoch 450 | Loss       0.42 | Correct   49 | Time/epoch 0.162s
Epoch 460 | Loss       0.23 | Correct   50 | Time/epoch 0.162s
Epoch 470 | Loss       1.51 | Correct   49 | Time/epoch 0.162s
Epoch 480 | Loss       1.33 | Correct   50 | Time/epoch 0.162s
Epoch 490 | Loss       0.71 | Correct   50 | Time/epoch 0.162s

# !python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET xor --RATE 0.05
Epoch   0 | Loss      13.42 | Correct   22 | Time/epoch 0.000s
Epoch  10 | Loss       2.22 | Correct   44 | Time/epoch 0.435s
Epoch  20 | Loss       2.19 | Correct   45 | Time/epoch 0.382s
Epoch  30 | Loss       4.21 | Correct   45 | Time/epoch 0.362s
Epoch  40 | Loss       3.14 | Correct   46 | Time/epoch 0.373s
Epoch  50 | Loss       2.00 | Correct   44 | Time/epoch 0.369s
Epoch  60 | Loss       3.74 | Correct   48 | Time/epoch 0.362s
Epoch  70 | Loss       2.22 | Correct   46 | Time/epoch 0.356s
Epoch  80 | Loss       1.39 | Correct   47 | Time/epoch 0.366s
Epoch  90 | Loss       1.95 | Correct   48 | Time/epoch 0.362s
Epoch 100 | Loss       2.07 | Correct   48 | Time/epoch 0.358s
Epoch 110 | Loss       2.03 | Correct   49 | Time/epoch 0.365s
Epoch 120 | Loss       2.42 | Correct   49 | Time/epoch 0.362s
Epoch 130 | Loss       1.87 | Correct   49 | Time/epoch 0.359s
Epoch 140 | Loss       1.43 | Correct   49 | Time/epoch 0.356s
Epoch 150 | Loss       1.76 | Correct   49 | Time/epoch 0.361s
Epoch 160 | Loss       1.33 | Correct   49 | Time/epoch 0.359s
Epoch 170 | Loss       1.17 | Correct   49 | Time/epoch 0.357s
Epoch 180 | Loss       0.89 | Correct   49 | Time/epoch 0.361s
Epoch 190 | Loss       1.90 | Correct   50 | Time/epoch 0.359s
Epoch 200 | Loss       1.13 | Correct   49 | Time/epoch 0.357s
Epoch 210 | Loss       0.52 | Correct   50 | Time/epoch 0.356s
Epoch 220 | Loss       0.76 | Correct   50 | Time/epoch 0.359s
Epoch 230 | Loss       0.45 | Correct   49 | Time/epoch 0.357s
Epoch 240 | Loss       0.37 | Correct   49 | Time/epoch 0.356s
Epoch 250 | Loss       0.34 | Correct   50 | Time/epoch 0.359s
Epoch 260 | Loss       0.69 | Correct   50 | Time/epoch 0.358s
Epoch 270 | Loss       0.42 | Correct   50 | Time/epoch 0.356s
Epoch 280 | Loss       0.28 | Correct   50 | Time/epoch 0.356s
Epoch 290 | Loss       0.48 | Correct   50 | Time/epoch 0.358s
Epoch 300 | Loss       0.12 | Correct   49 | Time/epoch 0.357s
Epoch 310 | Loss       0.24 | Correct   50 | Time/epoch 0.356s
Epoch 320 | Loss       0.73 | Correct   50 | Time/epoch 0.358s
Epoch 330 | Loss       0.91 | Correct   50 | Time/epoch 0.357s
Epoch 340 | Loss       0.43 | Correct   50 | Time/epoch 0.356s
Epoch 350 | Loss       0.14 | Correct   49 | Time/epoch 0.356s
Epoch 360 | Loss       1.00 | Correct   50 | Time/epoch 0.357s
Epoch 370 | Loss       0.13 | Correct   50 | Time/epoch 0.356s
Epoch 380 | Loss       0.89 | Correct   50 | Time/epoch 0.355s
Epoch 390 | Loss       0.31 | Correct   50 | Time/epoch 0.357s
Epoch 400 | Loss       0.67 | Correct   50 | Time/epoch 0.356s
Epoch 410 | Loss       0.40 | Correct   50 | Time/epoch 0.355s
Epoch 420 | Loss       0.77 | Correct   50 | Time/epoch 0.356s
Epoch 430 | Loss       0.34 | Correct   50 | Time/epoch 0.356s
Epoch 440 | Loss       0.29 | Correct   50 | Time/epoch 0.356s
Epoch 450 | Loss       0.16 | Correct   50 | Time/epoch 0.355s
Epoch 460 | Loss       0.48 | Correct   50 | Time/epoch 0.357s
Epoch 470 | Loss       0.58 | Correct   50 | Time/epoch 0.356s
Epoch 480 | Loss       0.23 | Correct   50 | Time/epoch 0.356s
Epoch 490 | Loss       0.15 | Correct   50 | Time/epoch 0.357s
