#![feature(try_trait, nll)]

extern crate bit_vec;
extern crate env_logger;
#[macro_use]
extern crate failure;
extern crate fnv;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;
extern crate num_complex;
extern crate openqasm;
extern crate order_stat;
extern crate ordered_float;
extern crate permutohedron;
extern crate pyo3;
extern crate rand;
extern crate rand_chacha;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::str::FromStr;

use openqasm::Qasm;
use ordered_float::NotNaN;
use pyo3::types::exceptions::ValueError;

mod mapping;
mod solver;
pub mod topology;

const QUANTUM_REGISTER_NAME: &str = "qopter";

pub type Float = NotNaN<f64>;

pub trait Qubit {
    fn index(&self) -> usize;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysicalQubit(usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LogicalQubit(usize);

impl Qubit for PhysicalQubit {
    fn index(&self) -> usize {
        self.0
    }
}

impl Qubit for LogicalQubit {
    fn index(&self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cnot<Q> {
    pub control: Q,
    pub target: Q,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Bit(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Swap {
    pub from: PhysicalQubit,
    pub to: PhysicalQubit,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Unitary<Q> {
    pub name: String,
    pub parameters: Vec<Float>,
    pub target: Q,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Gate<Q> {
    Cnot(Cnot<Q>),
    Swap(Swap),
    Unitary(Unitary<Q>),
    Barrier(Vec<Q>),
    Measure(Q, Bit),
}

impl Gate<PhysicalQubit> {
    fn qubits(&self) -> Vec<PhysicalQubit> {
        use Gate::*;
        match self {
            Cnot(cnot) => {
                let first = cnot.control.min(cnot.target);
                let second = cnot.control.max(cnot.target);
                vec![first, second]
            }
            Swap(swap) => {
                let first = swap.from.min(swap.to);
                let second = swap.from.max(swap.to);
                vec![first, second]
            }
            Unitary(u) => vec![u.target],
            _ => vec![],
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Fail)]
#[fail(display = "Parse Error")]
pub struct ParseError;

pub(crate) fn parse_topology_lines<
    'a,
    S: AsRef<str>,
    E: std::error::Error + Send + Sync + 'static,
    I: Iterator<Item = Result<S, E>>,
>(
    mut lines: I,
) -> Result<topology::ConnectionGraph, failure::Error> {
    let (num_physical_qubits, num_connections) = {
        let line = lines.next().ok_or(ParseError)??;
        let mut it = line.as_ref().split(' ');
        let first = usize::from_str(it.next().ok_or(ParseError)?.trim())?;
        let second = usize::from_str(it.next().ok_or(ParseError)?.trim())?;
        (first, second)
    };
    let mut single_qubit_fidelities = vec![];
    let mut readout_fidelities = vec![];
    for _ in 0..num_physical_qubits {
        let line = lines.next().ok_or(ParseError)??;
        let mut it = line.as_ref().split(' ');
        let single = f64::from_str(it.next().ok_or(ParseError)?.trim())?;
        let readout = f64::from_str(it.next().ok_or(ParseError)?.trim())?;
        single_qubit_fidelities.push(single.into());
        readout_fidelities.push(readout.into());
    }

    let mut connection = topology::ConnectionGraph::new(
        num_physical_qubits,
        single_qubit_fidelities,
        readout_fidelities,
    );
    for _ in 0..num_connections {
        let line = lines.next().ok_or(ParseError)??;
        let mut it = line.as_ref().split(' ');
        let left = usize::from_str(it.next().ok_or(ParseError)?.trim())?;
        let right = usize::from_str(it.next().ok_or(ParseError)?.trim())?;
        let fidelity = f64::from_str(it.next().ok_or(ParseError)?.trim())?.into();
        connection.connect(PhysicalQubit(left), PhysicalQubit(right), fidelity);
    }

    Ok(connection)
}

pub fn parse_topology(filename: &str) -> Result<topology::ConnectionGraph, failure::Error> {
    let lines = BufReader::new(File::open(filename)?).lines();
    parse_topology_lines(lines)
}

pub fn parse_qasm(filename: &str) -> Result<Qasm, failure::Error> {
    use std::io::Read;

    let mut buffer = String::new();
    File::open(filename)?.read_to_string(&mut buffer)?;
    let qasm = openqasm::from_str(&buffer)?;
    Ok(qasm)
}

#[derive(Debug, Clone)]
pub struct Solution {
    pub initial: mapping::Mapping,
    pub last: mapping::Mapping,
    pub qasm: Qasm,
    pub fidelity: Float,
    pub variables_to_qopter: HashMap<(String, usize), usize>,
}

pub fn run_solve(
    topology: &topology::ConnectionGraph,
    initial_mappings: usize,
    edge_to_edge: bool,
    beams: usize,
    mut paths: usize,
    qasm: Qasm,
    seed: Option<u64>,
    random: bool,
) -> Solution {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaChaRng;

    let mut rng = match seed {
        Some(s) => {
            info!("qopter random seed: {:?}", s);
            ChaChaRng::seed_from_u64(s)
        }
        None => {
            let mut rng = rand::thread_rng();
            let s = rng.gen();
            info!("qopter random seed: {:?}", s);
            ChaChaRng::seed_from_u64(s)
        }
    };

    if !edge_to_edge {
        assert!(initial_mappings > 0);
    }
    assert!(beams > 0);
    assert!(paths > 0);

    use openqasm::Operation as Op;
    let initial;
    let last;
    let operations;
    let fidelity;
    let topology_mapping;
    {
        topology_mapping = {
            let mut m = HashMap::new();
            let mut start = 0;
            for qr in qasm.quantum_registers.iter() {
                m.insert(qr.name.clone(), start);
                start += qr.size;
            }
            m
        };
        let (measurement_mapping, measurement_inverse) = {
            let mut m = HashMap::new();
            let mut inv = vec![];
            let mut start = 0;
            for cr in qasm.classical_registers.iter() {
                m.insert(&cr.name, start);
                start += cr.size;
                for _ in 0..cr.size {
                    inv.push(&cr.name);
                }
            }
            (m, inv)
        };

        let logical_qubit =
            |q: openqasm::Qubit| -> LogicalQubit { LogicalQubit(topology_mapping[&q.0] + q.1) };
        let qubit = |q: PhysicalQubit| -> openqasm::Qubit {
            openqasm::Qubit(QUANTUM_REGISTER_NAME.into(), q.index())
        };

        let gates: Vec<_> = qasm
            .operations
            .into_iter()
            .map(|op| match op {
                Op::Unitary(name, parameters, target) => Gate::Unitary(Unitary {
                    name,
                    parameters: parameters
                        .into_iter()
                        .map(|x| Float::new(x as f64).unwrap())
                        .collect(),
                    target: logical_qubit(target),
                }),
                Op::Cx(control, target) => Gate::Cnot(Cnot {
                    control: logical_qubit(control),
                    target: logical_qubit(target),
                }),
                Op::Measure(qubit, bit) => Gate::Measure(
                    logical_qubit(qubit),
                    Bit(measurement_mapping[&bit.0] + bit.1),
                ),
                Op::Barrier(qs) => Gate::Barrier(qs.into_iter().map(logical_qubit).collect()),
            })
            .collect();

        let answer;
        // TODO: avoid clone
        let reverse_gates = {
            let mut gs = gates.clone();
            gs.reverse();
            gs
        };
        // TODO: avoid box allocation
        let mut mapper: Box<Iterator<Item = mapping::Mapping>> = {
            let mut mappings =
                mapping::RandomMapper::new(topology.size, initial_mappings, &mut rng)
                    .collect::<Vec<_>>();
            if edge_to_edge {
                mappings.extend(mapping::edge_to_edge_mapping(1, topology, &gates, &mut rng));
            }
            Box::new(mappings.into_iter())
        };
        answer = loop {
            // Reverse Traversal Trick
            // forward pass
            let forward_answers = solver::beam_solve(topology, &gates, mapper, beams, random, &mut rng);

            paths -= 1;
            if paths == 0 {
                break forward_answers
                    .into_iter()
                    .max_by_key(|s| s.scheduler.fidelity)
                    .unwrap();
            }

            // backward pass
            let backward_answers = solver::beam_solve(
                topology,
                &reverse_gates,
                forward_answers.into_iter().map(|s| s.current),
                beams,
                random,
                &mut rng,
            );
            mapper = Box::new(backward_answers.into_iter().map(|s| s.current));
        };

        initial = answer.initial;
        last = answer.current;
        fidelity = answer.scheduler.fidelity;

        operations = answer
            .scheduler
            .gates
            .into_iter()
            .flat_map(|g| match g {
                // TODO: using vec! is not a good idea since it requires dynamic allocation
                Gate::Unitary(unitary) => vec![Op::Unitary(
                    unitary.name,
                    unitary
                        .parameters
                        .into_iter()
                        .map(|x| x.into_inner() as f64)
                        .collect(),
                    qubit(unitary.target),
                )],
                Gate::Cnot(cnot) => vec![Op::Cx(qubit(cnot.control), qubit(cnot.target))],
                Gate::Measure(q, bit) => vec![Op::Measure(qubit(q), {
                    let name = measurement_inverse[bit.0];
                    let index = bit.0 - measurement_mapping[name];
                    openqasm::Bit(name.to_string(), index)
                })],
                Gate::Barrier(qs) => vec![Op::Barrier(qs.into_iter().map(qubit).collect())],
                Gate::Swap(swap) => {
                    // expand swap into three cnots
                    // currently assuming the coupling map is undirected,
                    // or each edge has the same error rate for both direction
                    vec![
                        Op::Cx(qubit(swap.from), qubit(swap.to)),
                        Op::Cx(qubit(swap.to), qubit(swap.from)),
                        Op::Cx(qubit(swap.from), qubit(swap.to)),
                    ]
                }
            })
            .collect::<Vec<_>>();
    }

    let mut includes = qasm.includes;
    if !includes.iter().any(|x| x == "qelib1.inc") {
        includes.push("qelib1.inc".into());
    }

    let variables_to_qopter = {
        let mut m = HashMap::new();
        for qr in qasm.quantum_registers.iter() {
            let start = topology_mapping.get(&qr.name).unwrap();
            for i in 0..qr.size {
                let p = initial.to_physical(LogicalQubit(start + i));
                m.insert((qr.name.to_string(), i), p.index());
            }
        }
        m
    };

    info!(
        "mergeable operations: {}",
        count_gates_on_same_qubits(topology.size, &operations)
    );

    Solution {
        initial,
        last,
        qasm: Qasm {
            quantum_registers: vec![openqasm::QuantumRegister {
                name: QUANTUM_REGISTER_NAME.into(),
                size: topology.size,
            }],
            classical_registers: qasm.classical_registers,
            operations,
            includes,
        },
        fidelity,
        variables_to_qopter,
    }
}

fn count_gates_on_same_qubits(size: usize, operations: &[openqasm::Operation]) -> usize {
    let mut last_target = vec![vec![]; size];
    let mut count = 0;
    for op in operations {
        use openqasm::Operation::*;
        match op {
            Unitary(_, _, target) => {
                assert_eq!(target.0, QUANTUM_REGISTER_NAME);
                let next_target = vec![target.1];
                if last_target[target.1] == next_target {
                    count += 1;
                } else {
                    last_target[target.1] = next_target;
                }
            }
            Cx(control, target) => {
                assert_eq!(control.0, QUANTUM_REGISTER_NAME);
                assert_eq!(target.0, QUANTUM_REGISTER_NAME);
                let min = control.1.min(target.1);
                let max = control.1.max(target.1);
                let next_target = vec![min, max];

                if last_target[control.1] == next_target && last_target[target.1] == next_target {
                    count += 1;
                } else {
                    last_target[control.1] = next_target.clone();
                    last_target[target.1] = next_target;
                }
            }
            _ => {}
        }
    }

    count
}

use pyo3::prelude::*;

#[pymodinit]
fn libqopter(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "solve_file")]
    fn solve_file(
        _py: Python,
        qasm_file: &str,
        topology_file: &str,
        num_initial_mappings: usize,
        num_beams: usize,
        paths: usize,
        random: bool,
    ) -> PyResult<(String, f64)> {
        if let Err(_) = env_logger::try_init() {
            eprintln!("failed to initialize logger");
        }
        let qasm = parse_qasm(qasm_file).map_err(|e| ValueError::py_err(format!("{:?}", e)))?;
        let topology =
            parse_topology(topology_file).map_err(|e| ValueError::py_err(format!("{:?}", e)))?;
        let solution = run_solve(
            &topology,
            num_initial_mappings,
            false,
            num_beams,
            paths,
            qasm,
            None,
            random,
        );
        // eprintln!("{:?}", solution);

        Ok((solution.qasm.to_string(), solution.fidelity.into_inner()))
    }

    #[pyfn(m, "solve")]
    fn solve(
        _py: Python,
        qasm: &str,
        single_fidelity: Vec<f64>,
        cnot_fidelity: Vec<(usize, usize, f64)>,
        readout_fidelity: Vec<f64>,
        num_initial_mappings: usize,
        num_beams: usize,
        paths: usize,
        edge_to_edge: bool,
        random: bool,
    ) -> PyResult<(String, f64, HashMap<(String, usize), usize>)> {
        if let Err(_) = env_logger::try_init() {
            eprintln!("failed to initialize logger");
        }
        let qasm = openqasm::from_str(qasm).map_err(|e| ValueError::py_err(e.to_string()))?;

        assert!(single_fidelity.len() == readout_fidelity.len());
        let mut topology = topology::ConnectionGraph::new(
            single_fidelity.len(),
            single_fidelity
                .into_iter()
                .map(|x| Float::new(x).unwrap())
                .collect(),
            readout_fidelity
                .into_iter()
                .map(|x| Float::new(x).unwrap())
                .collect(),
        );
        for (q1, q2, fidelity) in cnot_fidelity.into_iter() {
            topology.connect(
                PhysicalQubit(q1),
                PhysicalQubit(q2),
                Float::new(fidelity).unwrap(),
            );
        }

        let solution = run_solve(
            &topology,
            num_initial_mappings,
            edge_to_edge,
            num_beams,
            paths,
            qasm,
            None,
            random,
        );

        Ok((
            solution.qasm.to_string(),
            solution.fidelity.into_inner(),
            solution.variables_to_qopter,
        ))
    }

    Ok(())
}
