## Task 3.2
### Diagnostic Script
```
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
```
## Task 3.4

============================================================
    Size |        CPU Time |        GPU Time |  Speedup
------------------------------------------------------------
      64 |         3.81 ms |         6.84 ms |     0.56x
     128 |        15.69 ms |        14.79 ms |     1.06x
     256 |        95.90 ms |        55.11 ms |     1.74x
     512 |       958.38 ms |       217.57 ms |     4.40x
    1024 |          7.75 s |          1.23 s |     6.32x
============================================================
<img width="1453" alt="Screenshot 2024-11-19 at 12 30 11 PM" src="https://github.com/user-attachments/assets/1d886697-3712-478b-bfcd-4b8364c22baf">

## Task 3.5
### Simple Dataset - GPU
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
<img width="545" alt="Screenshot 2024-11-18 at 8 57 19 PM" src="https://github.com/user-attachments/assets/ae719e95-9f62-4ba5-8ea3-ef023c8a2752">

### Simple Dataset - CPU
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
<img width="541" alt="Screenshot 2024-11-18 at 8 57 39 PM" src="https://github.com/user-attachments/assets/01edec0c-df5f-40ec-83ef-1bbf2243a3e0">

### Split Dataset - GPU
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
<img width="537" alt="Screenshot 2024-11-18 at 8 58 12 PM" src="https://github.com/user-attachments/assets/4ecf91ec-5916-4a51-ba74-36a5e2dd6286">

### Split Dataset - CPU
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
<img width="543" alt="Screenshot 2024-11-18 at 8 58 31 PM" src="https://github.com/user-attachments/assets/cbed8639-95bb-443d-ae65-bb4a63d29f06">

### XOR Dataset - GPU
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
<img width="547" alt="Screenshot 2024-11-18 at 8 58 56 PM" src="https://github.com/user-attachments/assets/7702952b-fb7f-4953-9e75-16a2a9ac7e0e">

### XOR Dataset - CPU
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
<img width="535" alt="Screenshot 2024-11-18 at 8 59 13 PM" src="https://github.com/user-attachments/assets/e1ebdf3c-b96f-4192-871f-e780983e148d">

### Big Model - XOR Dataset - CPU - 200 Layers
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET xor --RATE 0.05
<img width="537" alt="Screenshot 2024-11-18 at 8 59 38 PM" src="https://github.com/user-attachments/assets/a4adde0e-821c-40ea-8913-a00faa1f5998">

### Big Model - XOR Dataset - GPU - 200 Layers
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET xor --RATE 0.05
<img width="541" alt="Screenshot 2024-11-19 at 12 51 28 PM" src="https://github.com/user-attachments/assets/066d9788-0f47-45c4-8ac7-4437c6306e8a">
