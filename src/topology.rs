use super::{Float, PhysicalQubit, Qubit, Swap};
use std::collections::HashMap;

#[derive(Debug)]
pub struct ConnectionGraph {
    pub size: usize,
    pub connections: Vec<HashMap<PhysicalQubit, Float>>,
    pub single_fidelity: Vec<Float>,
    pub readout_fidelity: Vec<Float>,
    pub link_strength: Vec<Float>,
    pub maximum_single_fidelity: Float,
    pub maximum_readout_fidelity: Float,
    pub maximum_cnot_fidelity: Float,
}

impl ConnectionGraph {
    pub(crate) fn new(
        size: usize,
        single_fidelity: Vec<Float>,
        readout_fidelity: Vec<Float>,
    ) -> Self {
        ConnectionGraph {
            size,
            connections: vec![HashMap::new(); size],
            maximum_single_fidelity: single_fidelity.iter().cloned().max().unwrap(),
            maximum_readout_fidelity: readout_fidelity.iter().cloned().max().unwrap(),
            single_fidelity,
            readout_fidelity,
            link_strength: vec![0.0.into(); size],
            maximum_cnot_fidelity: std::f64::NEG_INFINITY.into(),
        }
    }

    pub(crate) fn connect(&mut self, q1: PhysicalQubit, q2: PhysicalQubit, fidelity: Float) {
        assert!(q1.index() < self.size);
        assert!(q2.index() < self.size);

        self.connections[q1.index()].insert(q2, fidelity);
        self.link_strength[q1.index()] += fidelity;
        self.link_strength[q2.index()] += fidelity;
        self.maximum_cnot_fidelity = self.maximum_cnot_fidelity.max(fidelity);
    }

    pub(crate) fn is_connected(&self, q1: PhysicalQubit, q2: PhysicalQubit) -> bool {
        assert!(q1.index() < self.size);
        assert!(q2.index() < self.size);

        self.connections[q1.index()].get(&q2).is_some()
    }

    pub(crate) fn single_fidelity(&self, q: PhysicalQubit) -> Float {
        assert!(q.index() < self.size);

        self.single_fidelity[q.index()]
    }

    pub(crate) fn readout_fidelity(&self, q: PhysicalQubit) -> Float {
        assert!(q.index() < self.size);

        self.readout_fidelity[q.index()]
    }

    pub(crate) fn cnot_fidelity(&self, q1: PhysicalQubit, q2: PhysicalQubit) -> Float {
        assert!(q1.index() < self.size);
        assert!(q2.index() < self.size);
        assert!(
            self.is_connected(q1, q2),
            "{} and {} are not connected",
            q1.index(),
            q2.index()
        );

        self.connections[q1.index()][&q2]
    }

    // calculate the path with highest sucess probability for each pair of qubits
    // we use Warshall-Floyd algorithm to compute the maximum path
    // complexity: O(Q^3)
    pub(crate) fn maximum_paths(&self) -> Vec<Vec<(Vec<PhysicalQubit>, Float)>> {
        let mut ret = vec![vec![(vec![], std::f64::NEG_INFINITY.into()); self.size]; self.size];

        for (i, connection) in self.connections.iter().enumerate() {
            for (&PhysicalQubit(j), &cost) in connection.iter() {
                ret[i][j] = (vec![PhysicalQubit(i)], cost);
            }
        }

        for (i, row) in ret.iter_mut().enumerate() {
            row[i] = (vec![], 0.0.into());
        }

        for k in 0..self.size {
            for i in 0..self.size {
                for j in 0..self.size {
                    if ret[i][j].1 < ret[i][k].1 + ret[k][j].1 {
                        let mut new_path = ret[i][k].0.clone();
                        new_path.extend(ret[k][j].0.iter());
                        ret[i][j] = (new_path, ret[i][k].1 + ret[k][j].1);
                    }
                }
            }
        }

        for row in ret.iter_mut() {
            for (j, cell) in row.iter_mut().enumerate() {
                cell.0.push(PhysicalQubit(j));
            }
        }

        ret
    }

    pub(crate) fn find_optimal_mitm_swap(
        &self,
        paths: &[Vec<(Vec<PhysicalQubit>, Float)>],
    ) -> Vec<Vec<(Vec<Swap>, Float)>> {
        let mut ret = vec![vec![(vec![], 0.0.into()); self.size]; self.size];
        for i in 0..self.size {
            for j in 0..self.size {
                let mut max_prob = std::f64::NEG_INFINITY.into();
                let mut max_swaps = vec![];
                if i != j {
                    for from in 0..self.size {
                        for (&PhysicalQubit(to), &cost) in self.connections[from].iter() {
                            // i -> from, j -> to
                            let total_prob1 = (paths[i][from].1 + paths[j][to].1) * 3.0 + cost;
                            // i -> to, j -> from
                            let total_prob2 = (paths[i][to].1 + paths[j][from].1) * 3.0 + cost;

                            if total_prob1 > max_prob && total_prob1 > total_prob2 {
                                max_prob = total_prob1;
                                let i_iter = paths[i][from].0.windows(2).map(|w| Swap {
                                    from: w[0],
                                    to: w[1],
                                });
                                let j_iter = paths[j][to].0.windows(2).map(|w| Swap {
                                    from: w[0],
                                    to: w[1],
                                });
                                max_swaps = i_iter.chain(j_iter).collect();
                            } else if total_prob2 > max_prob && total_prob2 > total_prob1 {
                                max_prob = total_prob1;
                                let i_iter = paths[i][to].0.windows(2).map(|w| Swap {
                                    from: w[0],
                                    to: w[1],
                                });
                                let j_iter = paths[j][from].0.windows(2).map(|w| Swap {
                                    from: w[0],
                                    to: w[1],
                                });
                                max_swaps = i_iter.chain(j_iter).collect();
                            }
                        }
                    }
                }
                ret[i][j] = (max_swaps, max_prob);
            }
        }
        ret
    }
}

#[test]
fn test_connection() {
    use mapping::Mapping;
    use LogicalQubit;

    let error_rate: Vec<Float> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        .into_iter()
        .map(Into::into)
        .collect();
    let mut connection = ConnectionGraph::new(6, error_rate.clone(), error_rate);

    connection.connect(PhysicalQubit(0), PhysicalQubit(1), 0.9_f64.ln().into());
    connection.connect(PhysicalQubit(1), PhysicalQubit(0), 0.9_f64.ln().into());

    connection.connect(PhysicalQubit(1), PhysicalQubit(2), 0.8_f64.ln().into());
    connection.connect(PhysicalQubit(2), PhysicalQubit(1), 0.8_f64.ln().into());

    connection.connect(PhysicalQubit(0), PhysicalQubit(3), 0.7_f64.ln().into());
    connection.connect(PhysicalQubit(3), PhysicalQubit(0), 0.7_f64.ln().into());

    connection.connect(PhysicalQubit(1), PhysicalQubit(4), 0.6_f64.ln().into());
    connection.connect(PhysicalQubit(4), PhysicalQubit(1), 0.6_f64.ln().into());

    connection.connect(PhysicalQubit(2), PhysicalQubit(5), 0.5_f64.ln().into());
    connection.connect(PhysicalQubit(5), PhysicalQubit(2), 0.5_f64.ln().into());

    connection.connect(PhysicalQubit(3), PhysicalQubit(4), 0.4_f64.ln().into());
    connection.connect(PhysicalQubit(4), PhysicalQubit(3), 0.4_f64.ln().into());

    connection.connect(PhysicalQubit(4), PhysicalQubit(5), 0.3_f64.ln().into());
    connection.connect(PhysicalQubit(5), PhysicalQubit(4), 0.3_f64.ln().into());

    let maximum_paths = connection.maximum_paths();
    let swaps = connection.find_optimal_mitm_swap(&maximum_paths);

    let true_maximum_paths = vec![
        vec![
            (vec![0], 0.0),
            (vec![0, 1], 0.9_f64.ln()),
            (vec![0, 1, 2], (0.9_f64 * 0.8).ln()),
            (vec![0, 3], 0.7_f64.ln()),
            (vec![0, 1, 4], (0.9_f64 * 0.6).ln()),
            (vec![0, 1, 2, 5], (0.9_f64 * 0.8 * 0.5).ln()),
        ],
        vec![
            (vec![1, 0], 0.9_f64.ln()),
            (vec![1], 0.0),
            (vec![1, 2], 0.8_f64.ln()),
            (vec![1, 0, 3], (0.9_f64 * 0.7).ln()),
            (vec![1, 4], 0.6_f64.ln()),
            (vec![1, 2, 5], (0.8_f64 * 0.5).ln()),
        ],
        vec![
            (vec![2, 1, 0], (0.8_f64 * 0.9).ln()),
            (vec![2, 1], 0.8_f64.ln()),
            (vec![2], 0.0),
            (vec![2, 1, 0, 3], (0.8_f64 * 0.9 * 0.7).ln()),
            (vec![2, 1, 4], (0.8_f64 * 0.6).ln()),
            (vec![2, 5], 0.5_f64.ln()),
        ],
        vec![
            (vec![3, 0], 0.7_f64.ln()),
            (vec![3, 0, 1], (0.7_f64 * 0.9).ln()),
            (vec![3, 0, 1, 2], (0.7_f64 * 0.9 * 0.8).ln()),
            (vec![3], 0.0),
            (vec![3, 4], 0.4_f64.ln()),
            (vec![3, 0, 1, 2, 5], (0.7_f64 * 0.9 * 0.8 * 0.5).ln()),
        ],
        vec![
            (vec![4, 1, 0], (0.6_f64 * 0.9).ln()),
            (vec![4, 1], 0.6_f64.ln()),
            (vec![4, 1, 2], (0.6_f64 * 0.8).ln()),
            (vec![4, 3], 0.4_f64.ln()),
            (vec![4], 0.0),
            (vec![4, 5], 0.3_f64.ln()),
        ],
        vec![
            (vec![5, 2, 1, 0], (0.5_f64 * 0.8 * 0.9).ln()),
            (vec![5, 2, 1], (0.5_f64 * 0.8).ln()),
            (vec![5, 2], 0.5_f64.ln()),
            (vec![5, 2, 1, 0, 3], (0.5_f64 * 0.8 * 0.9 * 0.7).ln()),
            (vec![5, 4], 0.3_f64.ln()),
            (vec![5], 0.0),
        ],
    ];

    for i in 0..6 {
        for j in 0..6 {
            assert_eq!(
                maximum_paths[i][j]
                    .0
                    .iter()
                    .map(|x| x.0)
                    .collect::<Vec<_>>(),
                true_maximum_paths[i][j].0
            );
            // nearly equal
            assert!((maximum_paths[i][j].1.into_inner() - true_maximum_paths[i][j].1) < 0.0001);

            println!(
                "{} <-> {}: {:?}, {}",
                i,
                j,
                swaps[i][j].0,
                swaps[i][j].1.into_inner()
            );

            if i != j {
                // nearly equal
                assert!((swaps[i][j].1 - swaps[j][i].1).into_inner() < 0.0001);

                let mut mapping = Mapping::from_to_physical(vec![0, 1, 2, 3, 4, 5]);
                for Swap { from, to } in swaps[i][j].0.iter() {
                    mapping.swap(*from, *to);
                }

                println!("{:?}", mapping);
                assert!(connection.is_connected(
                    mapping.to_physical(LogicalQubit(i)),
                    mapping.to_physical(LogicalQubit(j))
                ));
            }
        }
    }
}

#[test]
fn test_tokyo() -> Result<(), failure::Error> {
    use mapping::Mapping;
    use rand::seq::IteratorRandom;
    use rand::Rng;
    use LogicalQubit;

    #[derive(Debug)]
    enum Never {};

    impl std::fmt::Display for Never {
        fn fmt(&self, _: &mut std::fmt::Formatter) -> std::fmt::Result {
            unreachable!()
        }
    }

    impl std::error::Error for Never {}

    let file = include_str!("../tokyo.error.topology");
    let mut graph = ::parse_topology_lines::<_, Never, _>(file.lines().map(Ok))?;

    let mut rng = rand::thread_rng();

    // randomly add edges
    {
        let edges = (0..graph.size).flat_map(|x| (x..graph.size).map(move |y| (x, y)));
        let num = rng.gen_range(graph.size, graph.size * (graph.size - 1) / 2);

        for (from, to) in edges.choose_multiple(&mut rng, num) {
            let log_prob = rng.gen_range(0.0_f64, 1.0_f64).ln();
            graph.connect(
                PhysicalQubit(from),
                PhysicalQubit(to),
                Float::new(log_prob).unwrap(),
            );
            graph.connect(
                PhysicalQubit(to),
                PhysicalQubit(from),
                Float::new(log_prob).unwrap(),
            );
        }
    }

    println!("{:?}", graph);

    let paths = graph.maximum_paths();
    let swaps = graph.find_optimal_mitm_swap(&paths);

    for i in 0..6 {
        for j in 0..6 {
            println!(
                "{} <-> {}: {:?}, {}",
                i,
                j,
                swaps[i][j].0,
                swaps[i][j].1.into_inner()
            );

            if i != j {
                // nearly equal
                assert!((swaps[i][j].1 - swaps[j][i].1).into_inner() < 0.0001);

                let mut mapping = Mapping::from_to_physical((0..graph.size).collect());
                for Swap { from, to } in swaps[i][j].0.iter() {
                    mapping.swap(*from, *to);
                }

                println!("{:?}", mapping);
                assert!(graph.is_connected(
                    mapping.to_physical(LogicalQubit(i)),
                    mapping.to_physical(LogicalQubit(j))
                ));
            }
        }
    }

    Ok(())
}
