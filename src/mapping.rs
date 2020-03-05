use super::{topology, Gate, LogicalQubit, PhysicalQubit, Qubit};
use bit_vec::BitVec;
use rand::{seq::SliceRandom, Rng};
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static! {
    static ref ID: Mutex<usize> = Mutex::new(0);
}

#[derive(Clone, Debug, Hash)]
pub struct Mapping {
    pub logical_to_physical: Vec<PhysicalQubit>,
    pub physical_to_logical: Vec<LogicalQubit>,
    pub id: usize,
}

impl Mapping {
    pub fn from_to_physical(m: Vec<usize>) -> Self {
        let id = {
            let mut id_ref = ID.lock().unwrap();
            let id = *id_ref;
            *id_ref = id + 1;
            id
        };
        let mut physical_to_logical = vec![LogicalQubit(0); m.len()];
        for (i, &p) in m.iter().enumerate() {
            physical_to_logical[p] = LogicalQubit(i);
        }

        Mapping {
            logical_to_physical: m.iter().map(|&x| PhysicalQubit(x)).collect(),
            physical_to_logical,
            id,
        }
    }

    pub fn to_physical(&self, q: LogicalQubit) -> PhysicalQubit {
        self.logical_to_physical[q.index()]
    }

    pub fn to_logical(&self, q: PhysicalQubit) -> LogicalQubit {
        self.physical_to_logical[q.index()]
    }

    pub fn swap(&mut self, p1: PhysicalQubit, p2: PhysicalQubit) {
        let l1 = self.physical_to_logical[p1.index()];
        let l2 = self.physical_to_logical[p2.index()];

        self.logical_to_physical.swap(l1.index(), l2.index());
        self.physical_to_logical.swap(p1.index(), p2.index());
    }

    #[allow(dead_code)]
    fn ok(&self) {
        for (i, p) in self.logical_to_physical.iter().enumerate() {
            assert_eq!(
                self.physical_to_logical[p.index()].0,
                i,
                "{:?} vs {:?}",
                self.logical_to_physical,
                self.physical_to_logical
            );
        }
    }
}

/*
struct AllMapper(Mapping);

impl Iterator for AllMapper {
    type Item = Mapping;

    fn next(&mut self) -> Option<Mapping> {
        let ret = self.0.clone();
        if self.0.next_permutation() {
            Some(ret)
        } else {
            None
        }
    }
}

impl AllMapper {
    fn new(size: usize) -> Self {
        AllMapper((0..size).map(PhysicalQubit).collect())
    }
}
*/

#[derive(Debug)]
struct VariableGraph {
    total_cnots: Vec<usize>,
    cnots: Vec<HashMap<LogicalQubit, usize>>,
}

impl VariableGraph {
    fn new(gates: &[Gate<LogicalQubit>]) -> Self {
        let mut total_cnots = vec![];
        let mut cnots = vec![];

        for g in gates.iter() {
            match g {
                Gate::Cnot(gate) => {
                    let control = gate.control;
                    let target = gate.target;

                    total_cnots.resize(control.index().max(target.index()) + 1, 0);
                    cnots.resize(control.index().max(target.index()) + 1, HashMap::new());

                    total_cnots[control.index()] += 1;
                    total_cnots[target.index()] += 1;

                    *cnots[control.index()].entry(target).or_insert(0) += 1;
                    *cnots[target.index()].entry(control).or_insert(0) += 1;
                }
                _ => {}
            }
        }

        VariableGraph { total_cnots, cnots }
    }
}

pub(crate) fn edge_to_edge_mapping<R: Rng>(
    _num: usize,
    connection: &topology::ConnectionGraph,
    gates: &[Gate<LogicalQubit>],
    rng: &mut R,
) -> impl Iterator<Item = Mapping> {
    let variable_graph = VariableGraph::new(gates);

    let mut current_mapping = HashMap::new();
    let mut candidates = BinaryHeap::new();
    let mut mapped = BitVec::from_elem(connection.size, false);

    let strongest_relation = variable_graph
        .cnots
        .iter()
        .enumerate()
        .flat_map(|(from, edges)| {
            edges
                .iter()
                .map(move |(to, num)| (LogicalQubit(from), *to, *num))
        })
        .max_by_key(|x| x.2)
        .unwrap();
    let strongest_connection = connection
        .connections
        .iter()
        .enumerate()
        .flat_map(|(from, edges)| {
            edges
                .iter()
                .map(move |(to, reliability)| (from, *to, *reliability))
        })
        .max_by_key(|x| x.2)
        .unwrap();

    current_mapping.insert(strongest_relation.0, PhysicalQubit(strongest_connection.0));
    current_mapping.insert(strongest_relation.1, strongest_connection.1);
    mapped.set(strongest_connection.0, true);
    mapped.set((strongest_connection.1).0, true);

    for (to, num) in variable_graph.cnots[strongest_relation.0.index()].iter() {
        candidates.push((num, strongest_relation.0, *to));
    }
    for (to, num) in variable_graph.cnots[strongest_relation.1.index()].iter() {
        candidates.push((num, strongest_relation.0, *to));
    }

    while !candidates.is_empty() && current_mapping.len() < variable_graph.total_cnots.len() {
        // sanity check
        assert_eq!(
            current_mapping.len() + mapped.iter().filter(|x| !x).count(),
            connection.size
        );

        let (_num, from, to) = candidates.pop().unwrap();

        if current_mapping.contains_key(&from) && current_mapping.contains_key(&to) {
            continue;
        } else if current_mapping.contains_key(&from) {
            // pick the best free cnot connection adjacent to current_mapping[from]
            // if there is no such connection, skip it
            let from_qubit = current_mapping.get(&from).unwrap();

            if let Some((to_qubit, _)) = connection.connections[from_qubit.0]
                .iter()
                .filter(|(q, _)| !mapped.get(q.0).unwrap())
                .max_by_key(|(_, f)| *f)
            {
                assert!(current_mapping.insert(to, *to_qubit).is_none());
                mapped.set(to_qubit.0, true);

                for (to, num) in variable_graph.cnots[from.index()].iter() {
                    candidates.push((num, from, *to));
                }
            }
        } else if current_mapping.contains_key(&to) {
            // pick the best free cnot connection adjacent to current_mapping[to]
            // if there is no such connection, skip it
            let to_qubit = current_mapping.get(&to).unwrap();

            if let Some((from_qubit, _)) = connection.connections[to_qubit.0]
                .iter()
                .filter(|(q, _)| !mapped.get(q.0).unwrap())
                .max_by_key(|(_, f)| *f)
            {
                assert!(current_mapping.insert(from, *from_qubit).is_none());
                mapped.set(from_qubit.0, true);

                for (from, num) in variable_graph.cnots[to.index()].iter() {
                    candidates.push((num, *from, to));
                }
            }
        } else {
            unreachable!()
        }
    }

    // sanity check
    assert_eq!(
        current_mapping.len() + mapped.iter().filter(|x| !x).count(),
        connection.size
    );

    // randomly assign non-assigned variables to qubits
    let mut non_assigned_qubits = {
        let mut qubits = mapped
            .iter()
            .enumerate()
            .filter(|(_, mapped)| !mapped)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        assert_eq!(current_mapping.len() + qubits.len(), connection.size);
        qubits.shuffle(rng);
        qubits
    };
    let to_physical = {
        let mut to_physical = vec![11111; connection.size];
        for i in 0..connection.size {
            if let Some(q) = current_mapping.get(&LogicalQubit(i)) {
                to_physical[i] = q.index();
            } else {
                to_physical[i] = non_assigned_qubits.pop().unwrap();
            }
        }
        to_physical
    };

    // sanity check
    assert!(to_physical.iter().find(|&&x| x == 11111).is_none());

    std::iter::once(Mapping::from_to_physical(to_physical))
}

pub(crate) struct RandomMapper<'r, R> {
    allocate_num: usize,
    current: Vec<usize>,
    rng: &'r mut R,
}

impl<'r, R> RandomMapper<'r, R> {
    pub(crate) fn new(qubit_num: usize, allocate_num: usize, rng: &'r mut R) -> Self {
        RandomMapper {
            allocate_num,
            current: (0..qubit_num).collect(),
            rng,
        }
    }
}

impl<'r, R: Rng> Iterator for RandomMapper<'r, R> {
    type Item = Mapping;

    fn next(&mut self) -> Option<Mapping> {
        if self.allocate_num == 0 {
            None
        } else {
            self.current.shuffle(self.rng);

            self.allocate_num -= 1;

            Some(Mapping::from_to_physical(self.current.clone()))
        }
    }
}
