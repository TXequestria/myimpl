use std::borrow::Borrow;

type HashMap<K,V> = std::collections::hash_map::HashMap<K,V,nohash::BuildNoHashHasher<usize>>;
type HashSet<K> = std::collections::hash_set::HashSet<K,nohash::BuildNoHashHasher<usize>>;


#[derive(Clone)]
struct Neighbours {
    to:HashSet<usize>,
    from:HashSet<usize>,
}

impl Neighbours {
    fn new() -> Self {
        Self {
            to:HashSet::with_hasher(nohash::BuildNoHashHasher::default()),
            from:HashSet::with_hasher(nohash::BuildNoHashHasher::default())
        }
    }
    fn with_capacity(capacity:usize) -> Self {
        if capacity == 0 {
           return Self::new();
        }
        Self {to:HashSet::with_capacity_and_hasher(capacity,nohash::BuildNoHashHasher::default()),
            from:HashSet::with_capacity_and_hasher(capacity,nohash::BuildNoHashHasher::default())
        }
    }
    fn shrink_to_fit(&mut self) {
        self.to.shrink_to_fit();
        self.from.shrink_to_fit();
    }
}

impl Default for Neighbours {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
struct Visited {
    visited_nodes:HashSet<usize>,
    unvisited_nodes:HashSet<usize>
}

impl From<&DirectedGraph> for Visited {
    fn from(value: &DirectedGraph) -> Self {
        let mut new_visited = Self::with_capacity(value.nodes_len());
        for (node,_) in value.nodes.iter() {
            new_visited.push_node(node);
        }
        new_visited.shrink_to_fit();
        new_visited
    }
}

impl From<DirectedGraph> for Visited {
    fn from(value: DirectedGraph) -> Self {
        Self::from(&value)
    }
}

impl Visited {
    fn new() -> Self {
        Self {visited_nodes:HashSet::with_hasher(nohash::BuildNoHashHasher::default()),
        unvisited_nodes:HashSet::with_hasher(nohash::BuildNoHashHasher::default())}
    }
    fn with_capacity(capacity:usize) -> Self {
        if capacity == 0 {
            return Self::new();
        }
        Self {visited_nodes:HashSet::with_capacity_and_hasher(capacity,nohash::BuildNoHashHasher::default()),
            unvisited_nodes:HashSet::with_hasher(nohash::BuildNoHashHasher::default())
        }
    }
    fn push_node(&mut self,node:&usize) {

        if self.visited_nodes.contains(node) {
            debug_assert!(!self.unvisited_nodes.contains(node));
            return;
        }

        self.unvisited_nodes.insert(*node);
    }
    fn visit(&mut self,node:&usize) {
        if !self.unvisited_nodes.contains(node) {
            return
        };
        if self.visited_nodes.contains(node) {
            debug_assert!(!self.unvisited_nodes.contains(node));
            return
        };
        //操作条件：unvisited里存在，且visited里不存在
        self.unvisited_nodes.remove(node);
        self.visited_nodes.insert(*node);
    }
    //None:不存在
    //Some(false):存在但并未访问
    //Some(true):存在且已经访问
    fn is_visited(&self,node:&usize) -> Option<bool> {
        if self.unvisited_nodes.contains(node) {
            debug_assert!(!self.visited_nodes.contains(node));
            return Some(false);
        }
        if self.visited_nodes.contains(node) {
            debug_assert!(!self.unvisited_nodes.contains(node));
            return Some(true)
        }
        None
    }
    fn shrink_to_fit(&mut self) {
        self.unvisited_nodes.shrink_to_fit();
        self.visited_nodes.shrink_to_fit();
    }
}

impl<A:Borrow<usize>> FromIterator<A> for Visited {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let size = match iter.size_hint() {
            (_,Some(higher)) => {higher},
            (lower,None) => {lower}
        };
        let mut v = Self::with_capacity(size);
        for elem in iter {
            v.push_node(elem.borrow());
        }
        v.shrink_to_fit();
        v
    }
}

impl<T:AsRef<[usize]>> From<T> for Visited {
    fn from(value: T) -> Self {
        let mut new_visit = Self::with_capacity(value.as_ref().len());
        for node in value.as_ref() {
            new_visit.push_node(node);
        }
        new_visit.shrink_to_fit();
        new_visit
    }
}

impl Default for Visited {
    fn default() -> Self {
        Self::new()
    }
}

struct InDegreeMaps {
    nonzero:HashMap<usize,usize>,
    zeros:HashSet<usize>,
    visited:HashSet<usize>
}

impl InDegreeMaps {
    fn shrink_to_fit(&mut self) {
        self.nonzero.shrink_to_fit();
        self.zeros.shrink_to_fit();
        self.visited.shrink_to_fit();
    }

    fn is_finished(&self) -> bool {
        self.nonzero.is_empty() && self.zeros.is_empty()
    }

    //调整一个节点接下来指向节点的degree
    fn visit_and_change_degree(&mut self, node:&usize, graph:&DirectedGraph) {
        let Some(next_nodes) = graph.next_nodes(node)
            else {return};

        if self.is_visited(node) {
            return;
        }

        // move the current node into visited
        self.nonzero.remove(node);
        self.zeros.remove(node);
        self.visited.insert(*node);

        // lower all next nodes' degree by 1
        // O(E) for loop
        for node in next_nodes {
            if self.is_visited(node) {
                //skip visited next nodes
                continue
            }

            let degree = self.nonzero.get_mut(node)
                // not visited, not in nonzero, must in zeros
                .expect(&format!("A zero in degree node {node} has been pointed to"));

            if *degree > 1 {
                *degree -= 1;
            } else {
                self.nonzero.remove(node);
                self.zeros.insert(*node);
            }

        }

    }

    fn is_visited(&self,node:&usize) -> bool {
        if self.visited.contains(node) {
            // found in visted, must not in zero and nonzero
            debug_assert!(!self.nonzero.contains_key(node));
            debug_assert!(!self.zeros.contains(node));
            return true;
        }
        false
    }

    fn start_nodes(&self) -> HashSet<usize> {
        //寻找尚未visited，且入度为0的节点
        let mut set = self.zeros.clone();
        set.shrink_to_fit();
        set
    }

}

impl From<&DirectedGraph> for InDegreeMaps {
    fn from(value: &DirectedGraph) -> Self {
        let mut nonzero = HashMap::with_capacity_and_hasher(
            value.nodes_len(), nohash::BuildNoHashHasher::default()
        );
        let mut zeros = HashSet::with_capacity_and_hasher(
            value.nodes_len(), nohash::BuildNoHashHasher::default()
        );
        for (node,edges) in value.nodes.iter() {
            if edges.from.is_empty() {
                zeros.insert(*node);
            } else {
                nonzero.insert(*node,edges.from.len());
            }
        }
        zeros.shrink_to_fit();
        nonzero.shrink_to_fit();
        Self {nonzero,zeros,visited:HashSet::with_capacity_and_hasher(
            value.nodes_len(), nohash::BuildNoHashHasher::default()
        )}
    }
}

#[derive(Clone)]
pub struct DirectedGraph {
    edges_len:usize,
    nodes:HashMap<usize,Neighbours>
}

type Layers = Vec<HashSet<usize>>;
type Loop = HashMap<usize,usize>;

impl DirectedGraph {
    pub fn new() -> Self {
        Self {edges_len:0,nodes:HashMap::with_hasher(nohash::BuildNoHashHasher::default())}
    }
    pub fn with_capacity(capacity:usize) -> Self {
        if capacity == 0 {
            return Self::new();
        }
        Self {edges_len:0,nodes:HashMap::with_capacity_and_hasher(capacity, nohash::BuildNoHashHasher::default())}
    }
    pub fn shrink_to_fit(&mut self) {
        self.nodes.shrink_to_fit();
        for neighbours in self.nodes.values_mut() {
            neighbours.shrink_to_fit();
        }
    }
    pub fn nodes_len(&self) -> usize {
        self.nodes.len()
    }
    pub fn edges_len(&self) -> usize {
        self.edges_len
    }
    //返回一个当前节点的to节点的迭代器
    fn next_nodes(&self,node:&usize) -> Option<impl IntoIterator<Item = &usize>> {
        if self.nodes_len() == 0 {
            return None;
        }
        let neighbours = self.nodes.get(node)?.to.iter();
        Some(neighbours)
    }
    fn assert_cond(&self) {
        for neighbours in self.nodes.values() {
            for node in neighbours.to.iter() {
                if !self.nodes.contains_key(node) {
                    panic!("Node edges contained a non-existent node {node}")
                }
            }
            for node in neighbours.from.iter() {
                if !self.nodes.contains_key(node) {
                    panic!("Node edges contained a non-existent node {node}")
                }
            }
        }
    }
    fn assert_pair(&self,start:usize,end:usize) {
        let start_node = self.nodes.get(&start).expect(&format!("Start node {start} non-existent"));
        let end_node = self.nodes.get(&end).expect(&format!("End node {end} non-existent"));

        if !start_node.to.contains(&end) {
            panic!("Edge {start} -> {end} defined, but {end} is not in {start}'s to list");
        }
        if !end_node.from.contains(&start) {
            panic!("Edge {start} -> {end} defined, but {start} is not in {end}'s from list");
        }
    }
    pub fn push_pair_with_sizehint(&mut self,start:usize,end:usize,hint:usize) {
        if let Some(neighbours) = self.nodes.get_mut(&start) {
            neighbours.to.insert(end);
        }else{
            let mut new_edges = Neighbours::with_capacity(hint);
            new_edges.to.insert(end);
            self.nodes.insert(start,new_edges);
        }
        if let Some(neighbours) = self.nodes.get_mut(&end) {
            neighbours.from.insert(start);
        }else{
            let mut new_edges = Neighbours::with_capacity(hint);
            new_edges.from.insert(start);
            self.nodes.insert(end,new_edges);
        }

        self.edges_len += 1;

        #[cfg(debug_assertions)]
        self.assert_pair(start, end);
        #[cfg(debug_assertions)]
        self.assert_cond();
    }
    pub fn push_node_with_sizehint(&mut self,node:usize,hint:usize) {
        if self.nodes.contains_key(&node) {return;}
        // insert a node without adding edges
        self.nodes.insert(node, Neighbours::with_capacity(hint));
    }
    pub fn push_node(&mut self,node:usize) {
        self.push_node_with_sizehint(node, 0);
    }
    pub fn push_pair(&mut self,start:usize,end:usize) {
        self.push_pair_with_sizehint(start, end, 0);
    }
    pub fn dfs(&self,start_node:usize) -> Option<Vec<usize>> {
        if !self.nodes.contains_key(&start_node) {
            return None;
        }
        let mut visited:Visited = self.nodes.keys().collect();
        let mut stack = Vec::with_capacity(self.nodes_len());
        let mut order = Vec::with_capacity(self.nodes_len());
        stack.push(start_node);
        while stack.len() > 0 {
            let current = stack.pop()?;
            if visited.is_visited(&current)? {
                continue;
            }
            visited.visit(&current);
            order.push(current);
            for next_neighbours in self.nodes.get(&current)?.to.iter() {
                if !visited.is_visited(next_neighbours)? {
                    stack.push(*next_neighbours);
                }
            }
        }
        Some(order)
    }

    //visits all node and their next nodes, O(V + E)
    pub fn topological_sort(&self) -> Result<Layers,(Layers,Loop)> {
        let mut degree_map:InDegreeMaps = self.into();
        let mut layers = Vec::with_capacity(self.nodes_len());
        while !degree_map.is_finished() {
            let current_layer = degree_map.start_nodes();
            if current_layer.is_empty() {
                degree_map.shrink_to_fit();
                return Err((layers,degree_map.nonzero))
            }
            for node in current_layer.iter() {
                degree_map.visit_and_change_degree(node, self);
            }
            layers.push(current_layer)
        }
        layers.shrink_to_fit();
        Ok(layers)
    }
}

impl<A:Borrow<(usize,usize)>> FromIterator<A> for DirectedGraph {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let size = match iter.size_hint() {
            (_,Some(higher)) => {higher},
            (lower,None) => {lower}
        };
        let mut new_graph = Self::with_capacity(size);
        for pair in iter {
            let (start,end) = pair.borrow();
            new_graph.push_pair_with_sizehint(*start, *end,size);
        }
        new_graph.shrink_to_fit();
        new_graph
    }
}

impl<T> From<T> for DirectedGraph 
    where T:AsRef<[(usize,usize)]>
{
    fn from(value: T) -> Self {
        let mut new_graph = Self::with_capacity(value.as_ref().len());
        for (start,end) in value.as_ref() {
            new_graph.push_pair_with_sizehint(*start, *end,value.as_ref().len());
        }
        new_graph.shrink_to_fit();
        new_graph
    }
}

pub struct UnDirectedGraph {
    edges_len:usize,
    adjacency_list:HashMap<usize,HashSet<usize>>
}

impl Default for UnDirectedGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl UnDirectedGraph {
    pub fn edges_len(&self) -> usize {
        self.edges_len
    }
    pub fn nodes_len(&self) -> usize {
        self.adjacency_list.len()
    }
    pub fn is_empty(&self) -> bool {
        if self.nodes_len() == 0 {
            debug_assert!(self.edges_len() == 0);
            return true;
        }
        return false;
    }
    pub fn new() -> Self {
        Self {
            edges_len:0,
            adjacency_list:HashMap::with_hasher(
                nohash::BuildNoHashHasher::default()
            )
        }
    }
    pub fn with_capacity(capacity:usize) -> Self {
        Self {
            edges_len:0,
            adjacency_list:HashMap::with_capacity_and_hasher(
                capacity,
                nohash::BuildNoHashHasher::default())
        }
    }
    // only push node, not adding edges
    fn push_node<B:Borrow<usize>>(&mut self,node:B) {
        let node = node.borrow();
        if self.adjacency_list.contains_key(node) {
            return;
        }
        let adj_nodes:HashSet<usize> = HashSet::with_capacity_and_hasher(
            self.adjacency_list.capacity(),
            nohash::BuildNoHashHasher::default());
        self.adjacency_list.insert(*node,adj_nodes);
    }
    fn push_edge<B:Borrow<(usize,usize)>>(&mut self,edge:B) {
        let (node1,node2) = edge.borrow();
        let size_estimation = self.adjacency_list.capacity();
        let mut is_edge_present= false;
        if let Some(adj_nodes) = self.adjacency_list.get_mut(node1) {
            // if an node2 is found in node1's adjacency list, turn is_edge_present to true;
            // adj_nodes.insert() return false, if node2 is found, which means edge already present
            if !adj_nodes.insert(*node2) {is_edge_present = true};
        }else {
            let mut adj_nodes:HashSet<usize> = HashSet::with_capacity_and_hasher(size_estimation,
                nohash::BuildNoHashHasher::default());
            adj_nodes.insert(*node2);
            self.adjacency_list.insert(*node1, adj_nodes);
        }
        // insert node2 into graph, and register node1 as its neighbour
        if let Some(adj_nodes) = self.adjacency_list.get_mut(node2) {
            // if an node1 is found in node2's adjacency list, turn is_edge_present to true;
            // adj_nodes.insert() return false, if node1 is found, which means edge already present
            if !adj_nodes.insert(*node1) {is_edge_present = true};
        }else {
            let mut adj_nodes:HashSet<usize> = HashSet::with_capacity_and_hasher(size_estimation,
                nohash::BuildNoHashHasher::default());
            adj_nodes.insert(*node1);
            self.adjacency_list.insert(*node2, adj_nodes);
        }
        // edge not present, increase edge count
        if !is_edge_present {
            self.edges_len += 1;
        }
    }
    pub fn shrink_to_fit(&mut self) {
        self.adjacency_list.shrink_to_fit();
        for v in self.adjacency_list.values_mut() {
            v.shrink_to_fit();
        }
    }
}

impl<T:AsRef<[(usize,usize)]>> From<T> for UnDirectedGraph {
    fn from(value: T) -> Self {
        let size_estimation = value.as_ref().len();
        let mut new_graph = Self::with_capacity(size_estimation);
        for edge in value.as_ref() {
            new_graph.push_edge(edge);
        }
        new_graph.shrink_to_fit();
        new_graph
    }
}

impl<B:Borrow<(usize,usize)>> FromIterator<B> for UnDirectedGraph {
    fn from_iter<T: IntoIterator<Item = B>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let size_estimation = match iter.size_hint() {
            (_,Some(n)) => {n},
            (n,None) => {n}
        };
        let mut new_graph = Self::with_capacity(size_estimation);
        for b in iter {
            new_graph.push_edge(b.borrow());
        }
        new_graph.shrink_to_fit();
        new_graph
    }
}

#[cfg(test)]
mod tests{
    use rand::{Rng, RngCore};

    use super::{DirectedGraph, HashSet, UnDirectedGraph};

    #[test]
    fn test_insert() {
        let mut nodes:Vec<usize> = vec![];
        let mut rng = rand::rng();
        for _ in 0..16 {
            nodes.push(rng.next_u64() as usize);
        }
        let mut edges:Vec<(usize,usize)> = vec![];
        for i in 0..nodes.len() - 1 {
            edges.push((nodes[i],nodes[i+1]))
        }
        let new_graph = DirectedGraph::from(&edges);
        let order = new_graph.dfs(edges[0].0).unwrap();
        println!("{:?}",order);
        assert_eq!(order,nodes);
    }
    #[test]
    fn test_new_undirected_graph() {
        let mut rng = rand::rng();
        let edge_len:usize = rng.random_range(1000..10000);
        let mut edges:std::collections::HashSet<(usize,usize)> = std::collections::HashSet::new();
        for _ in 0..edge_len {
            edges.insert((rng.random_range(0..114514),rng.random_range(0..114514)));
        }
        let isolated_nodes_len:usize = rng.random_range(100..1000);

        let mut new_graph:UnDirectedGraph = edges.iter().collect();
        for n in 0..isolated_nodes_len {new_graph.push_node(n)};

        assert!(new_graph.edges_len == edges.len())
    }
    #[test]
    fn test_layer_cycle() {
        let edges = [(1,3),(1,4),(2,4),(2,5),(3,6),(4,6),(7,4),(5,8),(6,9),(9,7),(8,9)];
        let graph:DirectedGraph = edges.into();

        let (layers,loops) = graph.topological_sort().unwrap_err();

        println!("{layers:?}");
        println!("loop nodes : {loops:?}");

    }
}