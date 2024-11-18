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
-------------------------------------------------------------------------|loop #ID
    def _map(                                                            | 
        out: Storage,                                                    | 
        out_shape: Shape,                                                | 
        out_strides: Strides,                                            | 
        in_storage: Storage,                                             | 
        in_shape: Shape,                                                 | 
        in_strides: Strides,                                             | 
    ) -> None:                                                           | 
        # TODO: Implement for Task 3.1.                                  | 
        # Simple parallel loop with explicit indices                     | 
        for i in prange(len(out)):---------------------------------------| #2
            out_index = np.zeros(MAX_DIMS, np.int32)---------------------| #0
            in_index = np.zeros(MAX_DIMS, np.int32)----------------------| #1
            to_index(i, out_shape, out_index)                            | 
            broadcast_index(out_index, out_shape, in_shape, in_index)    | 
            o = index_to_position(out_index, out_strides)                | 
            j = index_to_position(in_index, in_strides)                  | 
            out[o] = fn(in_storage[j])                                   | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #0, #1).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--2 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--2 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--2 (parallel)
   +--0 (serial)
   +--1 (serial)


 
Parallel region 0 (loop #2) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#2).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (174) is hoisted out of the 
parallel loop labelled #2 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (175) is hoisted out of the 
parallel loop labelled #2 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: in_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (208)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/jacksoncamp/Desktop/CornellTech/CS 5781/mod3-jacksoncamp42/minitorch/fast_ops.py (208) 
-----------------------------------------------------------------------|loop #ID
    def _zip(                                                          | 
        out: Storage,                                                  | 
        out_shape: Shape,                                              | 
        out_strides: Strides,                                          | 
        a_storage: Storage,                                            | 
        a_shape: Shape,                                                | 
        a_strides: Strides,                                            | 
        b_storage: Storage,                                            | 
        b_shape: Shape,                                                | 
        b_strides: Strides,                                            | 
    ) -> None:                                                         | 
        # TODO: Implement for Task 3.1.                                | 
        # Simple parallel loop with explicit indices                   | 
        for i in prange(len(out)):-------------------------------------| #6
            out_index = np.zeros(MAX_DIMS, np.int32)-------------------| #3
            a_index = np.zeros(MAX_DIMS, np.int32)---------------------| #4
            b_index = np.zeros(MAX_DIMS, np.int32)---------------------| #5
                                                                       | 
            # Get positions                                            | 
            to_index(i, out_shape, out_index)                          | 
            broadcast_index(out_index, out_shape, a_shape, a_index)    | 
            broadcast_index(out_index, out_shape, b_shape, b_index)    | 
                                                                       | 
            o = index_to_position(out_index, out_strides)              | 
            a = index_to_position(a_index, a_strides)                  | 
            b = index_to_position(b_index, b_strides)                  | 
                                                                       | 
            # Apply operation                                          | 
            out[o] = fn(a_storage[a], b_storage[b])                    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #6, #3, #4, #5).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--6 is a parallel loop
   +--3 --> rewritten as a serial loop
   +--4 --> rewritten as a serial loop
   +--5 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--6 (parallel)
   +--3 (parallel)
   +--4 (parallel)
   +--5 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--6 (parallel)
   +--3 (serial)
   +--4 (serial)
   +--5 (serial)


 
Parallel region 0 (loop #6) had 0 loop(s) fused and 3 loop(s) serialized as part
 of the larger parallel loop (#6).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (222) is hoisted out of the 
parallel loop labelled #6 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (223) is hoisted out of the 
parallel loop labelled #6 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: a_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (224) is hoisted out of the 
parallel loop labelled #6 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: b_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (262)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/jacksoncamp/Desktop/CornellTech/CS 5781/mod3-jacksoncamp42/minitorch/fast_ops.py (262) 
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
        for i in prange(len(out)):-------------------------------| #8
            out_index = np.zeros(MAX_DIMS, np.int32)-------------| #7
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
loop(s) (originating from loops labelled: #8, #7).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--8 is a parallel loop
   +--7 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--7 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--7 (serial)


 
Parallel region 0 (loop #8) had 0 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/jacksoncamp/Desktop/CornellTech/CS 
5781/mod3-jacksoncamp42/minitorch/fast_ops.py (274) is hoisted out of the 
parallel loop labelled #8 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
/Users/jacksoncamp/Desktop/CornellTech/CS 5781/mod1-jacksoncamp42/.venv/lib/python3.12/site-packages/numba/core/typed_passes.py:336: NumbaPerformanceWarning: 
The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.

To find out why, try turning on parallel diagnostics, see https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics for help.

File "minitorch/fast_ops.py", line 287:

def _tensor_matrix_multiply(
^

  warnings.warn(errors.NumbaPerformanceWarning(msg,
Traceback (most recent call last):
  File "/Users/jacksoncamp/Desktop/CornellTech/CS 5781/mod3-jacksoncamp42/project/parallel_check.py", line 39, in <module>
    tmm(*out.tuple(), *a.tuple(), *b.tuple())
  File "/Users/jacksoncamp/Desktop/CornellTech/CS 5781/mod3-jacksoncamp42/minitorch/fast_ops.py", line 334, in _tensor_matrix_multiply
    raise NotImplementedError("Need to implement for Task 3.2")
NotImplementedError: Need to implement for Task 3.2