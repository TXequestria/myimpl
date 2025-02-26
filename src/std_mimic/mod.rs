// my implementation to mimic some std functions



// allocating memory and prevent fragmentation. 
// uses a base libc crate to get brk()-ed or HeapAlloc()-ed raw memory

pub struct MyAllocator {}