# compress

The memory occupied by a TensorFlow dense tensor depends on its data type and shape.

For example, if we have a dense tensor of float32 data type with a shape of [100, 100], it would occupy 4 bytes per element, as float32 requires 4 bytes per element. Therefore, the total memory occupied by this tensor would be:

100 x 100 x 4 = 40,000 bytes = 40 KB

Input tensors 2D

Output tensors 1D