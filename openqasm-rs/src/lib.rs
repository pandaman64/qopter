#[macro_use]
extern crate nom;
extern crate failure;

use nom::types::CompleteStr;
use std::fmt;
use std::str::FromStr;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct QuantumRegister {
    pub name: String,
    pub size: usize,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ClassicalRegister {
    pub name: String,
    pub size: usize,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Qubit(pub String, pub usize);

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Bit(pub String, pub usize);

#[derive(Clone, PartialEq, Debug)]
pub enum Operation {
    Unitary(String, Vec<f64>, Qubit),
    Cx(Qubit, Qubit),
    Measure(Qubit, Bit),
    Barrier(Vec<Qubit>),
}

#[derive(Clone, PartialEq, Debug)]
pub struct Qasm {
    pub quantum_registers: Vec<QuantumRegister>,
    pub classical_registers: Vec<ClassicalRegister>,
    pub operations: Vec<Operation>,
    pub includes: Vec<String>,
}

#[derive(Clone, PartialEq, Debug)]
enum Statement {
    Qreg(QuantumRegister),
    Creg(ClassicalRegister),
    Op(Operation),
    Include(String),
}

#[derive(Clone, Debug)]
pub struct Error {
    row: usize,
    column: usize,
    source: String,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "parse error at {}:{}, message: {}",
            self.row, self.column, self.source
        )
    }
}

impl std::error::Error for Error {}

type Result<'a> = std::result::Result<Qasm, Error>;

pub fn from_str(input: &str) -> Result {
    use nom::Context::*;
    use nom::Err::*;

    program(CompleteStr(input))
        .map(|(left, qasm)| {
            assert_eq!(left.0, "");
            qasm
        })
        .map_err(|e| match e {
            Error(Code(s, kind)) | Failure(Code(s, kind)) => {
                let (parsed, _) = input.split_at(input.len() - s.len());
                let lines = parsed.lines();

                let mut row = 0;
                let mut last_line = None;
                for line in lines {
                    row += 1;
                    last_line = Some(line);
                }

                let column = last_line.map(|line| line.chars().count()).unwrap_or(0);
                let source = format!("{:?}", kind);

                self::Error {
                    row,
                    column,
                    source,
                }
            }
            Incomplete(_) => unreachable!(),
        })
}

impl fmt::Display for Qasm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "OPENQASM 2.0;")?;
        for filename in self.includes.iter() {
            writeln!(f, "include \"{}\";", filename)?;
        }
        for qreg in self.quantum_registers.iter() {
            writeln!(f, "qreg {}[{}];", qreg.name, qreg.size)?;
        }
        for creg in self.classical_registers.iter() {
            writeln!(f, "creg {}[{}];", creg.name, creg.size)?;
        }
        for op in self.operations.iter() {
            use Operation::*;
            match op {
                Unitary(name, parameters, target) if !parameters.is_empty() => {
                    write!(f, "{}(", name)?;

                    let mut first = true;
                    for p in parameters.iter() {
                        if !first {
                            write!(f, ",")?;
                        }
                        write!(f, "{}", p)?;
                        first = false;
                    }

                    writeln!(f, ") {}[{}];", target.0, target.1)?;
                }
                Unitary(name, _, target) => writeln!(f, "{} {}[{}];", name, target.0, target.1)?,
                Cx(control, target) => writeln!(
                    f,
                    "cx {}[{}],{}[{}];",
                    control.0, control.1, target.0, target.1
                )?,
                Measure(qubit, bit) => {
                    writeln!(f, "measure {}[{}]->{}[{}];", qubit.0, qubit.1, bit.0, bit.1)?
                }
                Barrier(qubits) => {
                    write!(f, "barrier ")?;
                    let mut first = true;
                    for q in qubits.iter() {
                        if !first {
                            write!(f, ",")?;
                        }
                        write!(f, "{}[{}]", q.0, q.1)?;
                        first = false;
                    }
                    writeln!(f, ";")?;
                }
            }
        }
        Ok(())
    }
}

pub fn to_string(qasm: &Qasm) -> String {
    qasm.to_string()
}

pub fn double(input: CompleteStr) -> nom::IResult<CompleteStr, f64> {
    flat_map!(input, call!(nom::recognize_float), parse_to!(f64))
}

macro_rules! w (
    ($i:expr, $($args:tt)*) => (
        {
            sep!($i, separator, $($args)*)
        }
    )
);

named!(program<CompleteStr, Qasm>,
    w!(do_parse!(
        tag_no_case!(&CompleteStr("OPENQASM"))
        >> tag_no_case!(&CompleteStr("2.0"))
        >> tag_no_case!(&CompleteStr(";"))
        >> statements: many1!(statement)
        >> eof!()
        >> (statements.into_iter()
            .fold(Qasm {
                quantum_registers: vec![],
                classical_registers: vec![],
                operations: vec![],
                includes: vec![],
            }, |mut qasm, statement| {
                match statement {
                    Statement::Qreg(qreg) => qasm.quantum_registers.push(qreg),
                    Statement::Creg(creg) => qasm.classical_registers.push(creg),
                    Statement::Op(op) => qasm.operations.push(op),
                    Statement::Include(filename) => qasm.includes.push(filename),
                }
                qasm
            }))
    ))
);
named!(statement<CompleteStr, Statement>,
    w!(terminated!(
        alt_complete!(
            map!(include, Statement::Include)
            | map!(qreg, Statement::Qreg)
            | map!(creg, Statement::Creg)
            | map!(operation, Statement::Op)
        ),
        tag_no_case!(&CompleteStr(";"))
    ))
);

named!(operation<CompleteStr, Operation>,
    w!(alt_complete!(
        cx
        | measure
        | barrier
        | unitary
    ))
);
named!(unitary<CompleteStr, Operation>,
    w!(do_parse!(
        name: ident
        >> params: opt!(complete!(w!(delimited!(
            tag_no_case!(&CompleteStr("(")),
            separated_list!(
                tag_no_case!(&CompleteStr(",")),
                parameter
            ),
            tag_no_case!(&CompleteStr(")"))
        ))))
        >> q: qubit
        >> (Operation::Unitary(name.to_string(), params.unwrap_or_else(|| vec![]), q))
    ))
);
named!(cx<CompleteStr, Operation>,
    w!(do_parse!(
        tag_no_case!(&CompleteStr("CX"))
        >> q1: qubit
        >> tag_no_case!(&CompleteStr(","))
        >> q2: qubit
        >> (Operation::Cx(q1, q2))
    ))
);
named!(measure<CompleteStr, Operation>,
    w!(do_parse!(
        tag_no_case!(&CompleteStr("measure"))
        >> q: qubit
        >> tag_no_case!(&CompleteStr("->"))
        >> c: bit
        >> (Operation::Measure(q, c))
    ))
);
named!(barrier<CompleteStr, Operation>,
    w!(do_parse!(
        tag_no_case!(&CompleteStr("barrier"))
        >> qubits: separated_list!(tag_no_case!(&CompleteStr(",")), qubit)
        >> (Operation::Barrier(qubits))
    ))
);

named!(parameter<CompleteStr, f64>,
    call!(term)
);
named!(term<CompleteStr, f64>,
    w!(do_parse!(
        lhs: factor
        >> rhs: opt!(pair!(one_of!("+-"), term))
        >> (match rhs {
            Some(('+', rhs)) => lhs + rhs,
            Some(('-', rhs)) => lhs - rhs,
            None => lhs,
            _ => unreachable!(),
        })
    ))
);
named!(factor<CompleteStr, f64>,
    w!(do_parse!(
        lhs: primary
        >> rhs: opt!(pair!(one_of!("*/"), factor))
        >> (match rhs {
            Some(('*', rhs)) => lhs * rhs,
            Some(('/', rhs)) => lhs / rhs,
            None => lhs,
            _ => unreachable!(),
        })
    ))
);

named!(primary<CompleteStr, f64>,
    w!(alt_complete!(
        paren
        | neg
        | value!(std::f64::consts::PI, tag_no_case!("pi"))
        | call!(double)
    ))
);
named!(paren<CompleteStr, f64>,
    w!(delimited!(
        tag_no_case!("("),
        parameter,
        tag_no_case!(")")
    ))
);
named!(neg<CompleteStr, f64>,
    w!(do_parse!(
        tag_no_case!("-")
        >> value: call!(primary)
        >> (-value)
    ))
);

named_args!(register<'a>(t: &'a str)<CompleteStr<'a>, (String, usize)>,
    w!(do_parse!(
        tag_no_case!(t)
        >> name: ident
        >> tag_no_case!(&CompleteStr("["))
        >> size: map_res!(nom::digit, |s: CompleteStr| FromStr::from_str(s.0))
        >> tag_no_case!(&CompleteStr("]"))
        >> (name.to_string(), size)
    ))
);
named!(qreg<CompleteStr, QuantumRegister>, map!(call!(register, "qreg"), |(name, size)| QuantumRegister {
    name,
    size,
}));
named!(creg<CompleteStr, ClassicalRegister>, map!(call!(register, "creg"), |(name, size)| ClassicalRegister {
    name,
    size,
}));

named!(include<CompleteStr, String>,
    w!(do_parse!(
        tag_no_case!(&CompleteStr("include"))
        >> filename: delimited!(tag_no_case!(&CompleteStr("\"")), take_until!(&CompleteStr("\"")), tag_no_case!(&CompleteStr("\"")))
        >> (filename.to_string())
    ))
);

named!(operand<CompleteStr, (String, usize)>,
    w!(do_parse!(
        name: ident
        >> tag_no_case!(&CompleteStr("["))
        >> index: map_res!(nom::digit, |s: CompleteStr| FromStr::from_str(s.0))
        >> tag_no_case!(&CompleteStr("]"))
        >> (name.to_string(), index)
    ))
);
named!(qubit<CompleteStr, Qubit>, map!(operand, |t| Qubit(t.0, t.1)));
named!(bit<CompleteStr, Bit>, map!(operand, |t| Bit(t.0, t.1)));

named!(comment<CompleteStr, CompleteStr>,
    recognize!(preceded!(
        tag_no_case!(&CompleteStr("//")),
        terminated!(
            take_until_either!(&CompleteStr("\r\n")),
            alt_complete!(eof!() | call!(nom::eol))
        )
    ))
);

named!(ident<CompleteStr, CompleteStr>, recognize!(pair!(nom::alpha, nom::alphanumeric0)));
named!(separator<CompleteStr, CompleteStr>, alt_complete!(comment | eat_separator!(" \t\r\n")));

#[test]
fn test_parameter() {
    use std::f64::consts::PI;
    assert_eq!(
        parameter(CompleteStr("(-(pi + 1) / pi - 1)")),
        Ok((CompleteStr(""), -(PI + 1.0) / PI - 1.0)),
    );
    assert_eq!(
        parameter(CompleteStr("(-(pi + 1) / pi - -1)")),
        Ok((CompleteStr(""), -(PI + 1.0) / PI - -1.0)),
    );
}

#[test]
fn test_program() {
    assert_eq!(
        program(CompleteStr(
            r#"

OPENQASM 2.0;
include "qelib1.inc";
// comment
qreg q[5] ;
creg c [4 ]; 
U(1.2, 3, 4.56) q[3]; 
u1(3) qqq[3] ;
U(7, 8, 9) quuu[2] ;
cx qu[2], q[3];
X q[6];
barrier q [1];
measure q [ 1 ] -> c [ 3 ];
"#
        )),
        Ok((
            CompleteStr(""),
            Qasm {
                quantum_registers: vec![QuantumRegister {
                    name: "q".to_string(),
                    size: 5
                },],
                classical_registers: vec![ClassicalRegister {
                    name: "c".to_string(),
                    size: 4,
                },],
                operations: vec![
                    Operation::Unitary(
                        "U".to_string(),
                        vec![1.2, 3.0, 4.56],
                        Qubit("q".to_string(), 3)
                    ),
                    Operation::Unitary("u1".to_string(), vec![3.0], Qubit("qqq".to_string(), 3)),
                    Operation::Unitary(
                        "U".to_string(),
                        vec![7.0, 8.0, 9.0],
                        Qubit("quuu".to_string(), 2)
                    ),
                    Operation::Cx(Qubit("qu".to_string(), 2), Qubit("q".to_string(), 3)),
                    Operation::Unitary("X".to_string(), vec![], Qubit("q".to_string(), 6)),
                    Operation::Barrier(vec![Qubit("q".to_string(), 1)]),
                    Operation::Measure(Qubit("q".to_string(), 1), Bit("c".to_string(), 3)),
                ],
                includes: vec!["qelib1.inc".to_string(),],
            }
        ))
    );
}

#[test]
fn test_str() {
    let input = r#"
OPENQASM 2.0;
include "qelib1.inc";
// comment
qreg q[5] ;
creg c [4 ]; 
U(1.2, 3, 4.56) q[3]; 
u1(3) qqq[3] ;
U(7, 8, 9) quuu[2] ;
cx qu[2], q[3];
X q[6];
barrier q [1];
measure q [ 1 ] -> c [ 3 ];
"#;
    let output = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[4];
U(1.2,3,4.56) q[3];
u1(3) qqq[3];
U(7,8,9) quuu[2];
cx qu[2],q[3];
X q[6];
barrier q[1];
measure q[1]->c[3];
"#;
    assert_eq!(to_string(&from_str(input).unwrap()), output);
}
#[test]
fn test_operation() {
    assert_eq!(
        operation(CompleteStr("U(1.2, 3, 4.56)q[3]")),
        Ok((
            CompleteStr(""),
            Operation::Unitary(
                "U".to_string(),
                vec![1.2, 3.0, 4.56],
                Qubit("q".to_string(), 3)
            )
        ))
    );
    assert_eq!(
        operation(CompleteStr("\nbarrier q[0], q[2], q[3]\n")),
        Ok((
            CompleteStr("\n"),
            Operation::Barrier(vec![
                Qubit("q".to_string(), 0),
                Qubit("q".to_string(), 2),
                Qubit("q".to_string(), 3),
            ])
        ))
    );
}

#[test]
fn test_statement() {
    assert_eq!(
        statement(CompleteStr("  U(0,1,-1)q[0];")),
        Ok((
            CompleteStr(""),
            Statement::Op(Operation::Unitary(
                "U".to_string(),
                vec![0.0, 1.0, -1.0,],
                Qubit("q".to_string(), 0)
            ))
        ))
    );
    assert_eq!(
        statement(CompleteStr("\tU(0,1,-1)q[100];// hoge\n")),
        Ok((
            CompleteStr("// hoge\n"),
            Statement::Op(Operation::Unitary(
                "U".to_string(),
                vec![0.0, 1.0, -1.0,],
                Qubit("q".to_string(), 100)
            ))
        ))
    );
    assert_eq!(
        statement(CompleteStr("\nU(0,1,-1)q[100]  ; // hoge\n")),
        Ok((
            CompleteStr(" // hoge\n"),
            Statement::Op(Operation::Unitary(
                "U".to_string(),
                vec![0.0, 1.0, -1.0,],
                Qubit("q".to_string(), 100)
            ))
        ))
    );
}

#[test]
fn test_comment() {
    assert_eq!(
        comment(CompleteStr("// hoge\n")),
        Ok((CompleteStr(""), CompleteStr("// hoge\n")))
    );
    assert!(comment(CompleteStr("hoge")).is_err());
}
