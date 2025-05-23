use std::{borrow::Borrow, ops::Deref};

type HashMap<K,V> = std::collections::hash_map::HashMap<K,V,nohash::BuildNoHashHasher<usize>>;
type HashSet<K> = std::collections::hash_set::HashSet<K,nohash::BuildNoHashHasher<usize>>;

#[derive(Clone, Copy)]
pub enum NodeOrEdge {
    Node(usize),
    Edge(usize,usize)
}

impl NodeOrEdge {
    pub fn node(node:usize) -> Self {
        Self::Node(node)
    }
    pub fn edge(start:usize,end:usize) -> Self {
        Self::Edge(start, end)
    }
}

pub trait AsNodeOrEdge {
    fn parse(&self) -> NodeOrEdge;
}

impl AsNodeOrEdge for NodeOrEdge {
    fn parse(&self) -> NodeOrEdge {
        *self
    }
}

impl AsNodeOrEdge for usize {
    fn parse(&self) -> NodeOrEdge {
        NodeOrEdge::Node(*self)
    }
}

impl AsNodeOrEdge for (usize,usize) {
    fn parse(&self) -> NodeOrEdge {
        NodeOrEdge::Edge(self.0, self.1)
    }
}

impl AsNodeOrEdge for (&usize,usize) {
    fn parse(&self) -> NodeOrEdge {
        NodeOrEdge::Edge(*self.0, self.1)
    }
}

impl AsNodeOrEdge for (usize,&usize) {
    fn parse(&self) -> NodeOrEdge {
        NodeOrEdge::Edge(self.0, *self.1)
    }
}

impl AsNodeOrEdge for (&usize,&usize) {
    fn parse(&self) -> NodeOrEdge {
        NodeOrEdge::Edge(*self.0, *self.1)
    }
}

impl AsNodeOrEdge for [usize;2] {
    fn parse(&self) -> NodeOrEdge {
        NodeOrEdge::Edge(self[0], self[1])
    }
}

impl AsNodeOrEdge for [usize;1] {
    fn parse(&self) -> NodeOrEdge {
        NodeOrEdge::Node(self[0])
    }
}

impl AsNodeOrEdge for [&usize;2] {
    fn parse(&self) -> NodeOrEdge {
        NodeOrEdge::Edge(*self[0], *self[1])
    }
}

impl AsNodeOrEdge for [&usize;1] {
    fn parse(&self) -> NodeOrEdge {
        NodeOrEdge::Node(*self[0])
    }
}

impl AsNodeOrEdge for &[usize] {
    fn parse(&self) -> NodeOrEdge {
        match self.len() {
            1 => NodeOrEdge::Node(self[0]),
            2 => NodeOrEdge::Edge(self[0], self[1]),
            _ => panic!("only slice with len 1 or 2 can be converted")
        }
    }
}

impl<T:AsNodeOrEdge + ?Sized> AsNodeOrEdge for &T {
    fn parse(&self) -> NodeOrEdge {
        (*self).parse()
    }
}

impl<T:AsNodeOrEdge + ?Sized> AsNodeOrEdge for Box<T> {
    fn parse(&self) -> NodeOrEdge {
        self.deref().parse()
    }
}

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

impl From<&UnDirectedGraph> for Visited {
    fn from(value: &UnDirectedGraph) -> Self {
        let mut new_visited = Self::with_capacity(value.nodes_len());
        for n in value.adjacency_list.keys() {
            new_visited.push_node(n);
        }
        new_visited
    }
}

impl From<UnDirectedGraph> for Visited {
    fn from(value: UnDirectedGraph) -> Self {
        Self::from(&value)
    }
}

impl From<&DirectedGraph> for Visited {
    fn from(value: &DirectedGraph) -> Self {
        let mut new_visited = Self::with_capacity(value.nodes_len());
        for node in value.nodes.keys() {
            new_visited.push_node(node);
        }
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
    fn is_finished(&self) -> bool {
        self.unvisited_nodes.len() == 0
    }
    /* fn shrink_to_fit(&mut self) {
        self.unvisited_nodes.shrink_to_fit();
        self.visited_nodes.shrink_to_fit();
    } */
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
        //v.shrink_to_fit();
        v
    }
}

impl<T:AsRef<[usize]>> From<T> for Visited {
    fn from(value: T) -> Self {
        let mut new_visit = Self::with_capacity(value.as_ref().len());
        for node in value.as_ref() {
            new_visit.push_node(node);
        }
        //new_visit.shrink_to_fit();
        new_visit
    }
}

impl Default for Visited {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Clone,Debug)]
struct DegreeMap {
    nonzero_in:HashMap<usize,usize>,
    zero_in:HashSet<usize>,
    nonzero_out:HashMap<usize,usize>,
    zero_out:HashSet<usize>,
    visited:HashSet<usize>
}

// first element is node id, second
impl DegreeMap {
    fn select_first_n(&self,n:usize) -> HashSet<usize> {

        if n == 0 {
            return [].into_iter().collect()
        }

        type NodeID = usize;
        type NodeInDegree = usize;
        type NodeOutDegree = usize;
        type Slot = Option<(NodeID,NodeInDegree,NodeOutDegree)>;
        let mut slots:Vec<Slot> = vec![None;n];

        for (node_id,in_degree) in self.nonzero_in.iter() {
            let out_degree = self.nonzero_out.get(node_id).unwrap_or(&0);
            'inner:for slot in slots.iter_mut() {
                //slot 不为空
                if let Some((slot_id,slot_in_degree,slot_out_degree)) = slot.as_mut() {
                    //寻找出度最大的节点
                    if out_degree > slot_out_degree {
                        *slot_id = *node_id; *slot_in_degree = *in_degree; *slot_out_degree = *out_degree;
                        break 'inner;
                    }
                    //出度一样大，选择入度大的节点
                    if out_degree == slot_out_degree && in_degree > slot_in_degree {
                        *slot_id = *node_id; *slot_in_degree = *in_degree;
                        break 'inner;
                    }
                // slot 为空，直接插入
                }else{
                    *slot = Some((*node_id,*in_degree,*out_degree));
                }
            }
        }

        slots.into_iter().filter_map(|slot| {
            slot.map(|(node_id,_,_)| node_id)
        }).collect()

    }

    fn shrink_to_fit(&mut self) {
        self.nonzero_in.shrink_to_fit();
        self.zero_in.shrink_to_fit();
        self.nonzero_out.shrink_to_fit();
        self.zero_out.shrink_to_fit();
        self.visited.shrink_to_fit();
    }

    fn is_finished(&self) -> bool {
        self.nonzero_in.is_empty() && 
        self.zero_in.is_empty() &&
        self.zero_out.is_empty() &&
        self.nonzero_out.is_empty()
    }

    //调整一个节点接下来指向节点的degree
    fn visit_and_change_degree(&mut self, node:&usize, graph:&DirectedGraph) {
        let Some(next_nodes) = graph.next_nodes(node) else {return};
        let Some(from_nodes) = graph.previous_nodes(node) else {return};

        if self.is_visited(node) {
            return;
        }

        //println!("currently visiting node {node}");
        // move the current node into visited

        self.nonzero_in.remove(node);
        self.zero_in.remove(node);
        self.nonzero_out.remove(node);
        self.zero_out.remove(node);

        self.visited.insert(*node);

        // lower all next nodes' in_degree by 1
        // O(E) for loop
        for node in next_nodes.iter() {
            if self.is_visited(node) {
                //skip visited next nodes
                continue
            }
            let in_degree = self.nonzero_in.get_mut(node)
                // not visited, not in nonzero, must in zeros
                .expect(&format!("A zero in degree node {node} has been pointed to"));

            if *in_degree > 1 {
                *in_degree -= 1;
            } else {
                self.nonzero_in.remove(node);
                self.zero_in.insert(*node);
            }
        }

        // lower all from nodes' out_degree by 1
        for node in from_nodes.iter() {
            if self.is_visited(node) {
                //skip visited next nodes
                continue
            }
            let out_degree = self.nonzero_out.get_mut(node)
                // not visited, not in nonzero, must in zeros
                .expect(&format!("A zero to degree node {node} has pointed to another node"));
            if *out_degree > 1 {
                *out_degree -= 1;
            } else {
                self.nonzero_out.remove(node);
                self.zero_out.insert(*node);
            }
        }

    }

    fn is_visited(&self,node:&usize) -> bool {
        if self.visited.contains(node) {
            // found in visted, must not in zero and nonzero
            debug_assert!(!self.nonzero_in.contains_key(node));
            debug_assert!(!self.zero_in.contains(node));
            debug_assert!(!self.nonzero_out.contains_key(node));
            debug_assert!(!self.zero_out.contains(node));
            return true;
        }
        false
    }

    fn start_nodes(&self) -> HashSet<usize> {
        //寻找尚未visited，且入度为0的节点
        let mut set = self.zero_in.clone();
        set.shrink_to_fit();
        set
    }

}

impl From<&DirectedGraph> for DegreeMap {
    fn from(map:&DirectedGraph) -> Self {
        let mut nonzero_in = HashMap::with_capacity_and_hasher(
        map.nodes_len(), nohash::BuildNoHashHasher::default()
        );
        let mut zero_in = HashSet::with_capacity_and_hasher(
         map.nodes_len(), nohash::BuildNoHashHasher::default()
        );
        let mut nonzero_out = nonzero_in.clone();
        let mut zero_out = zero_in.clone();
        for (node,edges) in map.nodes.iter() {
            if edges.to.is_empty() {
                zero_out.insert(*node);
            } else {
                nonzero_out.insert(*node,edges.to.len());
            }
            if edges.from.is_empty() {
                zero_in.insert(*node);
            } else {
                nonzero_in.insert(*node,edges.from.len());
            }
        }
        nonzero_in.shrink_to_fit();
        zero_in.shrink_to_fit();
        nonzero_out.shrink_to_fit();
        zero_out.shrink_to_fit();
        Self {nonzero_in,zero_in,nonzero_out,zero_out,visited:HashSet::with_capacity_and_hasher(
            map.nodes_len(), nohash::BuildNoHashHasher::default()
        )}
    }
}

#[derive(Clone)]
pub struct DirectedGraph {
    edges_len:usize,
    nodes:HashMap<usize,Neighbours>
}

pub type MigrationSetps = Vec<(bool,HashSet<usize>)>;

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
    fn next_nodes(&self,node:&usize) -> Option<&HashSet<usize>> {
        if self.nodes_len() == 0 {
            return None;
        }
        let neighbours = &self.nodes.get(node)?.to;
        Some(neighbours)
    }
    fn previous_nodes(&self,node:&usize) -> Option<&HashSet<usize>> {
        if self.nodes_len() == 0 {
            return None;
        }
        let neighbours = &self.nodes.get(node)?.from;
        Some(neighbours)
    }
    #[cfg(debug_assertions)]
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
    #[cfg(debug_assertions)]
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
    fn push_pair_with_sizehint(&mut self,start:usize,end:usize,hint:usize) {
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
    fn push_node_with_sizehint(&mut self,node:usize,hint:usize) {
        if self.nodes.contains_key(&node) {return;}
        // insert a node without adding edges
        self.nodes.insert(node, Neighbours::with_capacity(hint));
    }
    fn push_node(&mut self,node:usize) {
        self.push_node_with_sizehint(node, 0);
    }
    fn push_pair(&mut self,start:usize,end:usize) {
        self.push_pair_with_sizehint(start, end, 0);
    }

    pub fn push<P:AsNodeOrEdge>(&mut self,node_or_edge:P) {
        match node_or_edge.parse() {
            NodeOrEdge::Node(node) => self.push_node(node),
            NodeOrEdge::Edge(start, end) => self.push_pair(start, end),
        }
    }

    pub fn push_with_sizehint<P:AsNodeOrEdge>(&mut self,node_or_edge:P,hint:usize) {
        match node_or_edge.parse() {
            NodeOrEdge::Node(node) => self.push_node_with_sizehint(node,hint),
            NodeOrEdge::Edge(start, end) => self.push_pair_with_sizehint(start, end,hint),
        }
    }

    fn remove_node(&mut self,node:usize) {
        let Some(neighbours) = self.nodes.remove(&node) else {return};
        let removed_len = neighbours.from.len() + neighbours.to.len();
        self.edges_len -= removed_len;

        for pointed_tos in neighbours.to.iter() {
            if let Some(ne) = self.nodes.get_mut(pointed_tos) {
                ne.from.remove(&node);
            }
        }
        for pointed_froms in neighbours.from.iter() {
            if let Some(ne) = self.nodes.get_mut(pointed_froms) {
                ne.to.remove(&node);
            }
        }
        self.shrink_to_fit();
    }

    pub fn dfs(&self,start_node:usize) -> Option<Vec<usize>> {
        if !self.nodes.contains_key(&start_node) {
            return None;
        }
        let mut visited:Visited = self.nodes.keys().collect();
        let mut stack = Vec::with_capacity(self.nodes_len());
        let mut order = Vec::with_capacity(self.nodes_len());
        stack.push(start_node);
        while let Some(current) = stack.pop() {
            //let current = stack.pop()?;
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
    pub fn migrate_with_vacancy(&mut self,vacancy_len:usize) -> MigrationSetps {
        let mut degree_map:DegreeMap = (&*self).into();

        let mut layers = Vec::with_capacity(self.nodes_len());
        while !degree_map.is_finished() {

            //println!("{degree_map:?}");

            let mut need_move_to_vacancy = false;
            let mut current_layer = degree_map.start_nodes();
            if current_layer.is_empty() {
                if vacancy_len == 0 {panic!("cannot perform make-before-break when no vacancy band")}
                need_move_to_vacancy = true;
                current_layer = degree_map.select_first_n(vacancy_len);
            }
            for node in current_layer.iter() {
                degree_map.visit_and_change_degree(node, self);
                self.remove_node(*node);
            }
            layers.push((need_move_to_vacancy,current_layer))
        }
        layers.shrink_to_fit();
        layers
    }
}

impl<A:AsNodeOrEdge> FromIterator<A> for DirectedGraph {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let size = match iter.size_hint() {
            (_,Some(higher)) => {higher},
            (lower,None) => {lower}
        };
        let size = 0;
        let mut new_graph = Self::with_capacity(size);
        for node_or_edge in iter {
            match node_or_edge.parse() {
                NodeOrEdge::Node(n) => {
                    new_graph.push_node_with_sizehint(n, size);
                },
                NodeOrEdge::Edge(start,end ) => {
                    new_graph.push_pair_with_sizehint(start, end,size);
                }
            }
        }
        new_graph.shrink_to_fit();
        new_graph
    }
}

impl<A:AsNodeOrEdge> Extend<A> for DirectedGraph {
    fn extend<T: IntoIterator<Item = A>>(&mut self, iter: T) {
        for node_or_edge in iter {
            match node_or_edge.parse() {
                NodeOrEdge::Edge(start, end) => {
                    self.push_pair(start,end);
                },
                NodeOrEdge::Node(node) => {
                    self.push_node(node);
                }
            }
        }
    }
}

impl<T> From<T> for DirectedGraph 
    where T:AsRef<[(usize,usize)]>
{
    fn from(value: T) -> Self {
        let size = value.as_ref().len();
        let size = 0;
        let mut new_graph = Self::with_capacity(size);
        for (start,end) in value.as_ref() {
            new_graph.push_pair_with_sizehint(*start, *end,size);
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

    pub fn push<P:AsNodeOrEdge>(&mut self,node_or_edge:P) {
        match node_or_edge.parse() {
            NodeOrEdge::Node(node) => self.push_node(node),
            NodeOrEdge::Edge(start, end) => self.push_edge((start, end)),
        }
    }

    // only push node, not adding edges
    fn push_node(&mut self,node:usize) {
        if self.adjacency_list.contains_key(&node) {
            return;
        }
        let adj_nodes:HashSet<usize> = HashSet::with_capacity_and_hasher(
            self.adjacency_list.capacity(),
            nohash::BuildNoHashHasher::default());
        self.adjacency_list.insert(node,adj_nodes);
        //debug_assert!(self.assert_integrety());
    }
    fn push_edge(&mut self,edge:(usize,usize)) {
        let size_estimation = self.adjacency_list.capacity();
        let (node1,node2) = edge;
        let mut is_edge_present= false;
        if let Some(adj_nodes) = self.adjacency_list.get_mut(&node1) {
            // if an node2 is found in node1's adjacency list, turn is_edge_present to true;
            // adj_nodes.insert() return false, if node2 is found, which means edge already present
            if !adj_nodes.insert(node2) {is_edge_present = true};
        }else {
            let mut adj_nodes:HashSet<usize> = HashSet::with_capacity_and_hasher(size_estimation,
                nohash::BuildNoHashHasher::default());
            adj_nodes.insert(node2);
            self.adjacency_list.insert(node1, adj_nodes);
        }
        // insert node2 into graph, and register node1 as its neighbour
        if let Some(adj_nodes) = self.adjacency_list.get_mut(&node2) {
            // if an node1 is found in node2's adjacency list, turn is_edge_present to true;
            // adj_nodes.insert() return false, if node1 is found, which means edge already present
            if !adj_nodes.insert(node1) {is_edge_present = true};
        }else {
            let mut adj_nodes:HashSet<usize> = HashSet::with_capacity_and_hasher(size_estimation,
                nohash::BuildNoHashHasher::default());
            adj_nodes.insert(node1);
            self.adjacency_list.insert(node2, adj_nodes);
        }
        // edge not present, increase edge count
        if !is_edge_present {
            self.edges_len += 1;
        }
        //debug_assert!(self.assert_integrety());
    }

    pub fn shrink_to_fit(&mut self) {
        self.adjacency_list.shrink_to_fit();
        for v in self.adjacency_list.values_mut() {
            v.shrink_to_fit();
        }
    }
}

impl UnDirectedGraph {
    #[cfg(debug_assertions)]
    fn is_edge_exist<N1:Borrow<usize>,N2:Borrow<usize>>(&self,node1:N1,node2:N2) -> bool{
        let node1 = node1.borrow();
        let node2 = node2.borrow();
        let mut edge_1_to_2_exsits = false;
        if let Some(node1_adj) = self.adjacency_list.get(node1) {
            edge_1_to_2_exsits = node1_adj.contains(node2);
        }
        if node1 == node2 {
            return edge_1_to_2_exsits;
        }
        let mut edge_2_to_1_exsists = false;
        if let Some(node2_adj) = self.adjacency_list.get(node2) {
            edge_2_to_1_exsists = node2_adj.contains(node1);
        }
        // for undirected graph, an edge exists if a -> b and b -> a
        edge_1_to_2_exsits && edge_2_to_1_exsists
    }

    #[allow(unused)]
    #[cfg(debug_assertions)]
    fn assert_integrety(&self) {
        let mut edge_count = 0;
        let mut visited:Visited = self.into();
        for (node1,node1_adj) in &self.adjacency_list {
            if visited.is_finished() {break;}
            if let Some(true) = visited.is_visited(node1) {
                continue;
            }
            for node2 in node1_adj {
                // node2 is already checked as src node, which means all node2->* has been checked
                // so skip checking node1 -> node2
                if let Some(true) = visited.is_visited(node2) {
                    continue;
                }
                // we have node1, and node2 is in node1's adj list
                debug_assert!(self.is_edge_exist(node1,node2));
                edge_count += 1;
            }
            // all node1 -> * has been checked, mark node1 as done
            visited.visit(node1);
        }
        debug_assert_eq!(edge_count,self.edges_len)
    }
}

impl<T:AsRef<[(usize,usize)]>> From<T> for UnDirectedGraph {
    fn from(value: T) -> Self {
        let size_estimation = value.as_ref().len();
        let mut new_graph = Self::with_capacity(size_estimation);
        for edge in value.as_ref() {
            new_graph.push_edge(*edge);
        }
        //debug_assert!(new_graph.assert_integrety());
        new_graph.shrink_to_fit();
        new_graph
    }
}

impl<A:AsNodeOrEdge> Extend<A> for UnDirectedGraph {
    fn extend<T: IntoIterator<Item = A>>(&mut self, iter: T) {
        for node_or_edge in iter {
            match node_or_edge.parse() {
                NodeOrEdge::Edge(start, end) => {
                    self.push_edge((start,end));
                },
                NodeOrEdge::Node(node) => {
                    self.push_node(node);
                }
            }
        }
    }
}

impl<A:AsNodeOrEdge> FromIterator<A> for UnDirectedGraph {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let size_estimation = match iter.size_hint() {
            (_,Some(n)) => {n},
            (n,None) => {n}
        };
        let mut new_graph = Self::with_capacity(size_estimation);
        for node_or_edge in iter {
            match node_or_edge.parse() {
                NodeOrEdge::Edge(start, end) => {
                    new_graph.push_edge((start,end));
                },
                NodeOrEdge::Node(node) => {
                    new_graph.push_node(node);
                }
            }
        }
        new_graph.shrink_to_fit();
        //debug_assert!(new_graph.assert_integrety());
        new_graph
    }
}

#[cfg(test)]
mod tests{
    use rand::{Rng, RngCore};

    use crate::dsa::graph::AsNodeOrEdge;

    use super::{DirectedGraph, UnDirectedGraph};

    #[test]
    fn test_insert() {
        let mut nodes:Vec<usize> = vec![];
        let mut rng = rand::rng();
        for _ in 0..16 {
            nodes.push(rng.next_u64() as usize);
        }
        let mut edges:Vec<Box<dyn AsNodeOrEdge>> = vec![];
        for i in 0..nodes.len() - 1 {
            edges.push(Box::new((nodes[i],nodes[i+1])))
        }
        let new_graph:DirectedGraph = edges.iter().collect();
        let order = new_graph.dfs(nodes[0]).unwrap();
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
        for _ in 0..isolated_nodes_len {new_graph.push_node(rng.random_range(0..1145141919810))};

        assert!(new_graph.edges_len == edges.len());
        new_graph.assert_integrety()
    }
    #[test]
    fn test_layer_cycle() {
        let edges = [(1,3),(1,4),(2,4),(2,5),(3,6),(4,6),(7,4),(5,8),(6,9),(9,7),(8,9)];
        let mut graph:DirectedGraph = edges.into();

        let migration_steps = graph.migrate_with_vacancy(3);

        for (need_move_to_vacancy,nodes) in migration_steps {
            if need_move_to_vacancy {
                println!("Nodes forced move to vacancy: {nodes:?}")
            }else{
                println!("regularly moving: {nodes:?} ")
            }
        }

    }
}