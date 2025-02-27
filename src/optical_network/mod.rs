// mimics a multi-band optical network using undirected graphs
// studies the implications of bandwidth fragmentation
// and non-linear effects in optical networks

#![allow(unused_imports)]

use crate::dsa::graph::{UnDirectedGraph,DirectedGraph};
use crate::linear_algebra::matrix::{Matrix,MatrixError};

/* assumes a bunch of nodes in row major order
suppose a total of M nodes, each nodes have identical bandwidth range 0-N
then each row is 1 by N representing a node
total M rows representing all nodes
Matrix M by N */

/* rules, suppose a task takes bandwith 5 block, and goes through link A->B->C->D
then A[5] B[5] C[5] D[5] must be all occupied
same bandwith block number must be used in all accessed link nodes for the same task
is is called bandwith consistency constraint
so for the same task it takes up a whole column */