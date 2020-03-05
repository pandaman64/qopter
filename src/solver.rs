use super::mapping::Mapping;
use super::topology::ConnectionGraph;
use super::{Bit, Cnot, Float, Gate, LogicalQubit, PhysicalQubit, Qubit, Swap, Unitary};
use bit_vec::BitVec;
use std::cmp::{Eq, Ordering, PartialEq, PartialOrd};
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;

#[derive(Clone, Debug)]
struct Dependency {
    adjacent_list: Vec<HashSet<usize>>,
}

fn executability<Q: PartialEq + std::fmt::Debug>(gates: &[Gate<Q>]) -> (Dependency, Vec<usize>) {
    // TODO: handle barrier op
    fn commute<Q: PartialEq>(left: &Gate<Q>, right: &Gate<Q>) -> bool {
        use Gate::*;
        if left == right {
            return true;
        }
        match (left, right) {
            (Unitary(left), Unitary(right)) => left.target != right.target,
            (Unitary(left), Cnot(right)) => {
                left.target != right.control && left.target != right.target
            }
            (Cnot(left), Unitary(right)) => {
                left.control != right.target && left.target != right.target
            }
            (Cnot(left), Cnot(right)) => {
                left.control == right.control
                    || (left.control != right.target
                        && left.target != right.control
                        && left.target != right.target)
            }
            (_, Measure(_, _)) | (Measure(_, _), _) => false,
            _ => false,
        }
    }

    let mut adjacent_list = vec![HashSet::new(); gates.len()];
    let mut indegree = vec![0; gates.len()];

    for right in 0..gates.len() {
        for left in 0..right {
            if !commute(&gates[left], &gates[right]) {
                adjacent_list[left].insert(right);
                indegree[right] += 1;
            }
        }
    }

    (Dependency { adjacent_list }, indegree)
}

#[derive(Debug, Clone)]
pub(crate) struct Scheduler<'c> {
    pub(crate) fidelity: Float,
    pub(crate) gates: Vec<Gate<PhysicalQubit>>,
    connection: &'c ConnectionGraph,
}

impl<'c> Scheduler<'c> {
    fn new(connection: &'c ConnectionGraph) -> Self {
        Scheduler {
            fidelity: 0.0.into(),
            gates: vec![],
            connection,
        }
    }

    /*
    fn append_swap(&mut self, i: PhysicalQubit, j: PhysicalQubit) {
        self.gates.push(Gate::Swap(Swap { from: i, to: j }));
        self.fidelity += self.connection.swap_fidelity(i, j);
    }
    */

    fn append_cnot(&mut self, cnot: Cnot<PhysicalQubit>) {
        use Gate::Cnot;
        self.fidelity += self.connection.cnot_fidelity(cnot.control, cnot.target);
        self.gates.push(Cnot(cnot));
    }

    fn append_unitary(&mut self, unitary: Unitary<PhysicalQubit>) {
        use Gate::Unitary;
        self.fidelity += self.connection.single_fidelity(unitary.target);
        self.gates.push(Unitary(unitary));
    }

    fn append_measure(&mut self, q: PhysicalQubit, b: Bit) {
        self.fidelity += self.connection.readout_fidelity(q);
        self.gates.push(Gate::Measure(q, b));
    }

    fn append_swaps(&mut self, fidelity: Float, swaps: Vec<Swap>) {
        self.fidelity += fidelity;
        self.gates.extend(swaps.into_iter().map(Gate::Swap));
    }

    fn append_barrier(&mut self, qs: Vec<PhysicalQubit>) {
        self.gates.push(Gate::Barrier(qs));
    }
}

#[derive(Debug)]
pub(crate) struct State<'c> {
    pub(crate) initial: Mapping,
    pub(crate) current: Mapping,
    remaining_gates: BitVec,
    ready_gates: BitVec,
    indegree: Vec<usize>,
    pub(crate) scheduler: Scheduler<'c>,
    pub(crate) cost: Float,
    // TODO: use this hash to remove duplicates
    hash: u64,
}

impl<'c> State<'c> {
    #[allow(dead_code)]
    fn print(&self) {
        println!("State {{");
        println!("  initial mapping:");
        for (i, q) in self.initial.logical_to_physical.iter().enumerate() {
            println!("    {} -> {}", i, q.index());
        }
        println!("  current mapping:");
        for (i, q) in self.current.logical_to_physical.iter().enumerate() {
            println!("    {} -> {}", i, q.index());
        }
        //println!("  remaining: {:b}", self.remaining_gates);
        println!("  remaining gates:");
        for g in self.remaining_gates.iter() {
            println!("    {}", g);
        }
        println!("  ready gates:");
        for g in self.ready_gates.iter() {
            println!("    {}", g);
        }
        println!("  processed gates: [");
        for gate in self.scheduler.gates.iter() {
            println!("    {:?}", gate);
        }
        println!("  ]");
        println!(
            "  current fidelity: {:.2}%",
            self.scheduler.fidelity.exp() * 100.0
        );
        println!("}}");
    }

    fn hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};

        let mut hasher = fnv::FnvHasher::with_key(42);
        self.current.hash(&mut hasher);
        self.remaining_gates.hash(&mut hasher);
        self.scheduler.gates.hash(&mut hasher);
        self.cost.hash(&mut hasher);

        hasher.finish()
    }
}

fn cost_function(
    scheduler: &Scheduler,
    mapping: &Mapping,
    swaps: &[Vec<(Vec<Swap>, Float)>],
    remaining_gates: &BitVec,
    gates: &[Gate<LogicalQubit>],
) -> Float {
    //scheduler.fidelity
    let connection = scheduler.connection;
    let mut result = scheduler.fidelity;
    for i in remaining_gates
        .iter()
        .enumerate()
        .filter(|x| x.1)
        .map(|x| x.0)
    {
        use Gate::*;
        match &gates[i] {
            Unitary(gate) => {
                let target = mapping.to_physical(gate.target);

                result += connection.single_fidelity(target);
            }
            Cnot(gate) => {
                let control = mapping.to_physical(gate.control);
                let target = mapping.to_physical(gate.target);

                result += swaps[control.index()][target.index()].1;
            }
            Measure(target, _) => {
                let target = mapping.to_physical(*target);

                result += connection.readout_fidelity(target);
            }
            Barrier(_) => {}
            ref g => unreachable!("{:?}", g),
        }
    }
    result
}

impl<'c, 'g> PartialEq for State<'c> {
    fn eq(&self, other: &Self) -> bool {
        // self.scheduler.fidelity == other.scheduler.fidelity
        self.cost == other.cost
    }
}

impl<'c, 'g> Eq for State<'c> {}

impl<'c, 'g> PartialOrd for State<'c> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.cost.partial_cmp(&other.cost).map(Ordering::reverse)
    }
}

impl<'c> Ord for State<'c> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.cmp(&other.cost).reverse()
    }
}

#[derive(Debug)]
pub(crate) enum Beams<'c> {
    Diversity {
        /// total beam size
        size: usize,
        /// beam size per initial mapping
        size_per_mapping: usize,
        /// states
        inner_map: HashMap<usize, BinaryHeap<State<'c>>>,
    },
    Random {
        size: usize,
        states: Vec<State<'c>>,
    },
}

impl<'c> Beams<'c> {
    fn new(size: usize, size_per_mapping: usize) -> Self {
        Beams::Diversity {
            size,
            size_per_mapping,
            inner_map: HashMap::new(),
        }
    }

    fn new_random(size: usize) -> Self {
        Beams::Random {
            size,
            states: vec![],
        }
    }

    fn push(&mut self, v: State<'c>) {
        match *self {
            Beams::Diversity {
                ref mut inner_map,
                size,
                size_per_mapping,
            } => {
                let heap = inner_map
                    .entry(v.initial.id)
                    .or_insert(BinaryHeap::with_capacity(size + 1));
                heap.push(v);
                if heap.len() > size_per_mapping {
                    heap.pop().unwrap();
                }
            }
            Beams::Random {
                ref mut states,
                ..
            } => {
                states.push(v);
            },
        }
    }

    fn finish<'r, R: rand::Rng>(self, rng: &'r mut R) -> Vec<State<'c>> {
        match self {
            Beams::Diversity {
                inner_map,
                size,
                ..
            } => {
                let mut states = vec![];
                for (_, ss) in inner_map.into_iter() {
                    states.extend(ss.into_vec());
                }
                if states.len() > size {
                    order_stat::kth(&mut states, size);
                }
                states.truncate(size);
                states
            }
            Beams::Random {
                size,
                mut states,
            } => {
                use rand::prelude::SliceRandom;
                states.shuffle(rng);
                states.truncate(size);
                states
            }
        }
    }
}

pub(crate) fn beam_solve<'c, 'g, M: Iterator<Item = Mapping>, R: rand::Rng>(
    connection: &'c ConnectionGraph,
    gates: &'g [Gate<LogicalQubit>],
    mapper: M,
    beams: usize,
    random: bool,
    rng: &mut R,
) -> Vec<State<'c>> {
    eprintln!("random = {}", random);
    let maximum_paths = connection.maximum_paths();
    let swap_sequences = connection.find_optimal_mitm_swap(&maximum_paths);

    for (i, v) in swap_sequences.iter().enumerate() {
        for (j, (swaps, _quality)) in v.iter().enumerate() {
            eprint!("{} -> {} [label=\"", i, j);
            for swap in swaps {
                eprint!("({}, {}) ", swap.from.index(), swap.to.index())
            }
            eprintln!("\"];");
        }
    }
    //let swap_sequences = connection.find_best_swap_sequences();
    let (dependency, initial_indegree) = executability(&gates);
    let remaining_gates = BitVec::from_elem(gates.len(), true);
    let ready_gates = BitVec::from_fn(gates.len(), |i| initial_indegree[i] == 0);

    let build_beam = move || {
        if random {
            Beams::new_random(beams)
        } else {
            Beams::new(beams, (beams / 10).max(1))
        }
    };

    // TODO: parametrize
    let mut best_so_far = build_beam();
    for permutation in mapper {
        let scheduler = Scheduler::new(connection);
        best_so_far.push({
            let mut s = State {
                hash: 0,
                cost: cost_function(
                    &scheduler,
                    &permutation,
                    &swap_sequences,
                    &remaining_gates,
                    gates,
                ),
                initial: permutation.clone(),
                current: permutation,
                remaining_gates: remaining_gates.clone(),
                ready_gates: ready_gates.clone(),
                indegree: initial_indegree.clone(),
                scheduler,
            };
            s.hash = s.hash();
            s
        });
    }

    for _ in 0..gates.len() {
        // TODO: parametrize
        let mut next_best_so_far = build_beam();
        for state in best_so_far.finish(rng).iter() {
            for g in state
                .ready_gates
                .iter()
                .enumerate()
                .filter(|x| x.1)
                .map(|x| x.0)
            {
                let mut next = state.current.clone();
                let mut scheduler = state.scheduler.clone();
                match &gates[g] {
                    Gate::Unitary(gate) => {
                        scheduler.append_unitary(Unitary {
                            name: gate.name.clone(),
                            parameters: gate.parameters.clone(),
                            target: state.current.to_physical(gate.target),
                        });
                    }
                    Gate::Cnot(gate) => {
                        let control = state.current.to_physical(gate.control);
                        let target = state.current.to_physical(gate.target);
                        if connection.is_connected(control, target) {
                            scheduler.append_cnot(Cnot { control, target });
                        } else {
                            let sequences = &swap_sequences[control.index()][target.index()];
                            for Swap { from, to } in sequences.0.iter() {
                                next.swap(*from, *to);
                            }
                            scheduler.append_swaps(sequences.1, sequences.0.clone());

                            let control = next.to_physical(gate.control);
                            let target = next.to_physical(gate.target);
                            scheduler.append_cnot(Cnot { control, target });
                        }
                    }
                    Gate::Measure(q, b) => {
                        let q = state.current.to_physical(*q);
                        scheduler.append_measure(q, *b);
                    }
                    Gate::Barrier(qs) => {
                        let qs = qs.iter().map(|q| state.current.to_physical(*q)).collect();
                        scheduler.append_barrier(qs);
                    }
                    Gate::Swap(_) => unreachable!(),
                }

                let mut remaining_gates = state.remaining_gates.clone();
                let mut ready_gates = state.ready_gates.clone();
                let mut indegree = state.indegree.clone();
                remaining_gates.set(g, false);
                ready_gates.set(g, false);
                for &child in dependency.adjacent_list[g].iter() {
                    assert!(indegree[child] > 0);
                    indegree[child] -= 1;
                    if indegree[child] == 0 {
                        ready_gates.set(child, true);
                    }
                }
                next_best_so_far.push({
                    let mut s = State {
                        hash: 0,
                        cost: cost_function(
                            &scheduler,
                            &next,
                            &swap_sequences,
                            &remaining_gates,
                            gates,
                        ),
                        initial: state.initial.clone(),
                        current: next,
                        remaining_gates,
                        ready_gates,
                        indegree,
                        scheduler,
                    };
                    s.hash = s.hash();
                    s
                });
            }
        }

        best_so_far = next_best_so_far;
    }

    let states = best_so_far.finish(rng);
    assert!(states.iter().all(|s| s.remaining_gates.none()));
    states
}
