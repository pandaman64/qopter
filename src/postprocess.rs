use super::{Float, Gate, PhysicalQubit};
use num_complex::Complex64;
use pyo3::prelude::*;

fn err<T>(res: PyResult<T>, py: Python) -> PyResult<T> {
    match res {
        Ok(x) => Ok(x),
        Err(e) => {
            e.clone_ref(py).print(py);
            Err(e)
        }
    }
}

type GateMatrix = ndarray::ArrayBase<ndarray::OwnedRepr<Complex64>, ndarray::Dim<[usize; 2]>>;

fn gates_to_matrix(
    gates: &[Gate<PhysicalQubit>],
    first: PhysicalQubit,
    second: PhysicalQubit,
) -> GateMatrix {
    let mut gate_matrix = array![
        [Complex64::new(1.0, 0.0), 0.0.into(), 0.0.into(), 0.0.into()],
        [0.0.into(), 1.0.into(), 0.0.into(), 0.0.into()],
        [0.0.into(), 0.0.into(), 1.0.into(), 0.0.into()],
        [0.0.into(), 0.0.into(), 0.0.into(), 1.0.into()],
    ];

    let cnot1 = array![
        [Complex64::new(1.0, 0.0), 0.0.into(), 0.0.into(), 0.0.into()],
        [0.0.into(), 1.0.into(), 0.0.into(), 0.0.into()],
        [0.0.into(), 0.0.into(), 0.0.into(), 1.0.into()],
        [0.0.into(), 0.0.into(), 1.0.into(), 0.0.into()],
    ];

    let cnot2 = array![
        [Complex64::new(1.0, 0.0), 0.0.into(), 0.0.into(), 0.0.into()],
        [0.0.into(), 0.0.into(), 0.0.into(), 1.0.into()],
        [0.0.into(), 0.0.into(), 1.0.into(), 0.0.into()],
        [0.0.into(), 1.0.into(), 0.0.into(), 0.0.into()],
    ];

    for gate in gates.iter() {
        match gate {
            Gate::Cnot(cnot) => {
                if cnot.control == first && cnot.target == second {
                    gate_matrix = gate_matrix.dot(&cnot2);
                } else if cnot.control == second && cnot.target == first {
                    gate_matrix = gate_matrix.dot(&cnot1);
                } else {
                    unreachable!("invalid operands: {:?}", gate);
                }
            }
            Gate::Unitary(unitary) => {
                let single = calc_unitary_gate_matrix(&unitary.name, &unitary.parameters);
                let mat;
                if unitary.target == first {
                    mat = array![
                        [single[0][0], single[0][1], 0.0.into(), 0.0.into()],
                        [single[1][0], single[1][1], 0.0.into(), 0.0.into()],
                        [0.0.into(), 0.0.into(), single[0][0], single[0][1]],
                        [0.0.into(), 0.0.into(), single[1][0], single[1][1]],
                    ];
                } else if unitary.target == second {
                    mat = array![
                        [single[0][0], 0.0.into(), single[0][1], 0.0.into()],
                        [0.0.into(), single[0][0], 0.0.into(), single[0][1]],
                        [single[1][0], 0.0.into(), single[1][1], 0.0.into()],
                        [0.0.into(), single[1][0], 0.0.into(), single[1][1]],
                    ];
                } else {
                    unreachable!("invalid operands: {:?}", unitary);
                }

                gate_matrix = gate_matrix.dot(&mat);
            }
            Gate::Swap(s) => {
                gate_matrix = gate_matrix.dot(&array![
                    [1.0.into(), 0.0.into(), 0.0.into(), 0.0.into()],
                    [0.0.into(), 0.0.into(), 1.0.into(), 0.0.into()],
                    [0.0.into(), 1.0.into(), 0.0.into(), 0.0.into()],
                    [0.0.into(), 0.0.into(), 0.0.into(), 1.0.into()],
                ]);
            }
            g => unreachable!("unsupported gate: {:?}", g),
        }
    }

    gate_matrix
}

#[test]
fn test_gates_to_matrix() {
    let first = PhysicalQubit(0);
    let second = PhysicalQubit(1);
    let gates = vec![
        Gate::Unitary(super::Unitary {
            name: "u3".into(),
            parameters: vec![Float::new(std::f64::consts::PI).unwrap(), Float::new(0.0).unwrap(), Float::new(std::f64::consts::PI).unwrap()],
            target: second,
        }),
        Gate::Cnot(super::Cnot {
            control: first,
            target: second,
        })
    ];

    fn assert_nearly_eq(x: &GateMatrix, y: &GateMatrix) {
        for i in 0..4 {
            for j in 0..4 {
                assert!((x[(i, j)] - y[(i, j)]).norm() < 0.0001, "x[{0}, {1}] = {2}, y[{0}, {1}] = {3}", i, j, x[(i, j)], y[(i, j)]);
            }
        }
    }

    let calculated = gates_to_matrix(&gates, first, second);
    let truth = array![
        [0.0.into(), 0.0.into(), 1.0.into(), 0.0.into()],
        [0.0.into(), 1.0.into(), 0.0.into(), 0.0.into()],
        [1.0.into(), 0.0.into(), 0.0.into(), 0.0.into()],
        [0.0.into(), 0.0.into(), 0.0.into(), 1.0.into()],
    ];

    println!("{:?}", calculated);

    assert_nearly_eq(
        &calculated,
        &truth,
    );
}

pub(crate) fn merge_two_qubit_gates(
    gates: &[Gate<PhysicalQubit>],
    first: PhysicalQubit,
    second: PhysicalQubit,
) -> PyResult<Vec<Gate<PhysicalQubit>>> {
    let gate_matrix = gates_to_matrix(gates, first, second);
    println!("{:?}", gate_matrix);

    let guard = Python::acquire_gil();
    let py = guard.python();
    let numpy = py.import("numpy")?;
    let array = numpy.get("array")?;
    let reshape = numpy.get("reshape")?;
    let complex_py = py.eval("complex", None, None)?;

    let to_python_complex = |x: Complex64| complex_py.call1((x.re, x.im));

    let arr_py = array.call1((gate_matrix
        .as_slice()
        .unwrap()
        .iter()
        .map(|x| err(to_python_complex(*x), py))
        .collect::<PyResult<Vec<_>>>()?,))?;
    let mat_py = reshape.call1((arr_py, (4, 4)))?;

    println!("{:?}", mat_py);

    let mapper = err(py.import("qiskit.mapper"), py)?;
    let two_qubit_kak = err(mapper.get("two_qubit_kak"), py)?;

    let decomposed = err(two_qubit_kak
        .call1((mat_py, true)), py)?
        .cast_as::<pyo3::types::PyList>()?;
    
    for gate in decomposed.iter() {
        println!("{:?}", gate);
    }

    let gates = decomposed
        .iter()
        .map(|gate| {
            let gate = gate.cast_as::<pyo3::types::PyDict>()?;
            let name = gate.get_item("name").unwrap().extract::<String>()?;
            let args = gate.get_item("args").unwrap().extract::<Vec<usize>>()?;
            let parameters = gate
                .get_item("params")
                .unwrap()
                .extract::<Vec<f64>>()?
                .into_iter()
                .map(|x| Float::new(x).unwrap())
                .collect();

            let single_target = if args[0] == 0 {
                first
            } else if args[0] == 1 {
                second
            } else {
                unreachable!("two_qubit_kak returned invalid value {:?}", gate)
            };

            info!("{:?}, {:?}, {:?}", name, args, parameters);

            // TODO: parametersが変（u1なのに3つあったりする）
            // ↑これはQiskit側がゲート名に関わらずu3のパラメータを出してる
            Ok(match name.as_str() {
                "cx" => {
                    let control;
                    let target;
                    if args[0] == 0 && args[1] == 1 {
                        control = first;
                        target = second;
                    } else if args[0] == 1 && args[1] == 0 {
                        control = second;
                        target = first;
                    } else {
                        unreachable!("two_qubit_kak returned invalid value {:?}", gate);
                    }
                    Gate::Cnot(super::Cnot { control, target })
                }
                "id" | "u1" | "u2" | "u3" => Gate::Unitary(super::Unitary {
                    name: "u3".into(),
                    parameters,
                    target: single_target,
                }),
                _ => unreachable!("two_qubit_kak returned unsupported gate {:?}", gate),
            })
        })
        .collect::<PyResult<_>>()?;

    Ok(gates)
}

#[test]
fn test_merge_two_qubit_gates() -> PyResult<()> {
    let first = PhysicalQubit(0);
    let second = PhysicalQubit(1);
    /*let gates = vec![
        Gate::Cnot(super::Cnot {
            control: first,
            target: second,
        }),
        Gate::Cnot(super::Cnot {
            control: second,
            target: first,
        }),
        Gate::Cnot(super::Cnot {
            control: first,
            target: second,
        }),
    ];
    */
    let gates = vec![
        Gate::Unitary(super::Unitary {
            name: "u3".into(),
            parameters: vec![Float::new(std::f64::consts::PI).unwrap(), Float::new(0.0).unwrap(), Float::new(std::f64::consts::PI).unwrap()],
            target: first,
        }),
        Gate::Cnot(super::Cnot {
            control: first,
            target: second,
        }),
        Gate::Cnot(super::Cnot {
            control: second,
            target: first,
        }),
        Gate::Cnot(super::Cnot {
            control: first,
            target: second,
        }),
        Gate::Unitary(super::Unitary {
            name: "u3".into(),
            parameters: vec![Float::new(std::f64::consts::PI).unwrap(), Float::new(0.0).unwrap(), Float::new(std::f64::consts::PI).unwrap()],
            target: second,
        }),
    ];

    let gates = merge_two_qubit_gates(&gates, first, second)?;
    for gate in gates.iter() {
        println!("{:?}", gate);
    }

    let second_gates = merge_two_qubit_gates(&gates, first, second);
    for gate in second_gates {
        println!("{:?}", gate);
    }

    Ok(())
}

/// NOTE two_qubit_kak sometimes fails with 'incorrect result'
/// in such cases, we should give up merging two-qubit gates
#[test]
fn test_call_qiskit() -> PyResult<()> {
    let guard = Python::acquire_gil();
    let py = guard.python();
    let _qiskit = err(py.import("qiskit"), py)?;
    let np = err(py.import("numpy"), py)?;
    let mapper = err(py.import("qiskit.mapper"), py)?;
    let two_qubit_kak = err(mapper.get("two_qubit_kak"), py)?;
    let euler_angles_1q = err(mapper.get("euler_angles_1q"), py)?;
    let np_array = err(np.get("array"), py)?;

    assert!(two_qubit_kak.is_callable());
    assert!(euler_angles_1q.is_callable());

    let angles = err(
        euler_angles_1q.call1((err(
            np_array.call1((vec![vec![1.0, 0.0], vec![0.0, 1.0]],)),
            py,
        )?,)),
        py,
    )?;
    println!("{:?}", angles);

    let decomposed = err(two_qubit_kak.call1(
        (err(np_array.call1((vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ],)), py)?,)
    ), py)?;
    println!("{:?}", decomposed);

    Ok(())
}

/// https://github.com/Qiskit/qiskit-terra/blob/d063ab8b2aae277b7b2a17031d581ec26bccc364/qiskit/backends/aer/_simulatortools.py#L86
pub fn calc_unitary_gate_matrix(name: &str, parameters: &[Float]) -> [[Complex64; 2]; 2] {
    let zero = Float::new(0.0).unwrap();
    let (theta, phi, lam) = match name {
        s if s == "u3" || s == "U" => (parameters[0], parameters[1], parameters[2]),
        s if s == "u2" => (
            Float::new(std::f64::consts::PI / 2.0).unwrap(),
            parameters[0],
            parameters[1],
        ),
        s if s == "u1" => (zero, zero, parameters[0]),
        s if s == "id" => (zero, zero, zero),
        s => unimplemented!("unimplemented gate: {}", s),
    };

    let cos_theta_2 = Complex64::new((theta / 2.0).into_inner().cos(), 0.0);
    let sin_theta_2 = Complex64::new((theta / 2.0).into_inner().sin(), 0.0);

    [
        [
            cos_theta_2,
            -Complex64::new(0.0, lam.into_inner()).exp() * sin_theta_2,
        ],
        [
            Complex64::new(0.0, phi.into_inner()).exp() * sin_theta_2,
            Complex64::new(0.0, (phi + lam).into_inner()).exp() * cos_theta_2,
        ],
    ]
}

#[test]
fn test_match_unitary_gate_matrix() -> PyResult<()> {
    use rand::seq::SliceRandom;
    use rand::Rng;

    let guard = Python::acquire_gil();
    let py = guard.python();
    let tools = py.import("qiskit.backends.aer._simulatortools")?;
    let single_gate_matrix = tools.get("single_gate_matrix")?;

    let gate_names = vec!["u1", "u2", "u3"];
    let mut rng = rand::thread_rng();

    // FIXME: calling python fuctions many times causes SEGV only on this project...
    for _ in 0..1000 {
        let random_gate_name = *gate_names.choose(&mut rng).unwrap();
        let params;
        let gate = match random_gate_name {
            s if s == "u1" => {
                params = vec![rng.gen::<f64>()];
                err(single_gate_matrix.call1((s, params.clone())), py)?
            }
            s if s == "u2" => {
                params = vec![rng.gen::<f64>(), rng.gen()];
                err(single_gate_matrix.call1((s, params.clone())), py)?
            }
            s if s == "u3" => {
                params = vec![rng.gen::<f64>(), rng.gen(), rng.gen()];
                err(single_gate_matrix.call1((s, params.clone())), py)?
            }
            _ => unreachable!(),
        };
        let ideal = {
            let mut ideal = [[Complex64::new(0.0, 0.0); 2]; 2];
            for i in 0..2 {
                for j in 0..2 {
                    let elem = err(gate.get_item((i, j)), py)?;
                    let real = err(err(elem.getattr("real"), py)?.extract(), py)?;
                    let imag = err(err(elem.getattr("imag"), py)?.extract(), py)?;
                    ideal[i][j] = Complex64::new(real, imag);
                }
            }
            ideal
        };
        let mine = calc_unitary_gate_matrix(
            random_gate_name,
            &params
                .iter()
                .map(|x| Float::new(*x).unwrap())
                .collect::<Vec<_>>(),
        );
        assert_eq!(ideal, mine);
    }

    Ok(())
}

#[test]
fn test_call_many_times() -> PyResult<()> {
    let guard = Python::acquire_gil();
    let py = guard.python();
    let tools = py.import("qiskit.backends.aer._simulatortools")?;
    let single_gate_matrix = tools.get("single_gate_matrix")?;
    let gate = "u1";
    let params = vec![0.0_f64];

    for i in 0..10 {
        println!("{}", i);
        err(single_gate_matrix.call1((gate, params.clone())), py)?;
    }

    Ok(())
}
