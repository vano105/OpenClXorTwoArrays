__kernel void xor(__global const bool *a, __global const bool *b, __global bool *c, unsigned int n) {
  const unsigned int index = get_global_id(0);

  if (index >= n)
    return;
  c[index] = a[index] ^ b[index];
}