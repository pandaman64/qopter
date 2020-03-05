#![feature(duration_as_u128)]

extern crate clap;
#[macro_use]
extern crate failure;
extern crate openqasm;
extern crate qopter;
extern crate rayon;

use clap::{App, Arg, SubCommand};
use openqasm::Qasm;
use qopter::*;
use rayon::prelude::*;
use std::str::FromStr;

#[derive(Debug)]
struct CommandLineOption {
    topology: topology::ConnectionGraph,
    qasm: Qasm,
    beams: usize,
    initial_mappings: usize,
    paths: usize,
    edge_to_edge: bool,
    is_etequal: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Fail)]
#[fail(display = "not enough arguments")]
struct NotEnoughArgumentsError;

fn command_line() -> Result<CommandLineOption, failure::Error> {
    let matches = App::new("QOPTER")
        .arg(
            Arg::with_name("topology")
                .long("topology")
                .value_name("TOPOLOGY_FILE")
                .help("file of topology"),
        )
        .arg(
            Arg::with_name("qasm")
                .long("qasm")
                .value_name("QASM_FILE")
                .help("file to compile"),
        )
        .arg(
            Arg::with_name("beams")
                .short("b")
                .long("beams")
                .value_name("NUM_BEAMS")
                .help("number of beams to use"),
        )
        .arg(
            Arg::with_name("mapping size")
                .short("s")
                .long("mapping_size")
                .value_name("NUM_MAPPING")
                .help("number of initial mappings"),
        )
        .arg(
            Arg::with_name("number of paths")
                .short("p")
                .long("path")
                .value_name("NUM_PATHS")
                .help("number of paths"),
        )
        .arg(
            Arg::with_name("edge-to-edge mapping")
                .short("e")
                .long("edge-to-edge")
                .value_name("EDGE_TO_EDGE")
                .help("use edge-to-edge mapping")
                .takes_value(false),
        )
        .subcommand(SubCommand::with_name("etequal"))
        .get_matches();

    let (topology, qasm) = match (matches.value_of("topology"), matches.value_of("qasm")) {
        (Some(topology), Some(qasm)) => {
            let topology = parse_topology(topology)?;
            let qasm = parse_qasm(qasm)?;
            (topology, qasm)
        }
        _ => return Err(NotEnoughArgumentsError.into()),
    };

    let beams = match matches.value_of("beams") {
        Some(s) => FromStr::from_str(s)?,
        None => 10,
    };

    let initial_mappings = match matches.value_of("mapping size") {
        Some(s) => FromStr::from_str(s)?,
        None => 0,
    };

    let paths = match matches.value_of("number of paths") {
        Some(s) => FromStr::from_str(s)?,
        None => 1,
    };

    Ok(CommandLineOption {
        topology,
        qasm,
        beams,
        initial_mappings,
        paths,
        edge_to_edge: matches.is_present("edge-to-edge mapping"),
        is_etequal: matches.subcommand_name() == Some("etequal"),
    })
}

fn main() -> Result<(), failure::Error> {
    let options = command_line()?;
    if !options.is_etequal {
        let solution = run_solve(
            &options.topology,
            options.initial_mappings,
            options.edge_to_edge,
            options.beams,
            options.paths,
            options.qasm,
            None,
            false,
        );

        println!("{:?}", solution);
        println!("{}", solution.qasm);
        println!("{}", solution.fidelity.into_inner().exp());
    } else {
        const STEP: usize = 50;
        const SHOTS: usize = 100;
        (1..options.initial_mappings)
            .step_by(STEP)
            .map(|x| std::iter::repeat(x).take(SHOTS).collect::<Vec<_>>())
            .flatten()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|&initial_mappings| {
                let start = std::time::Instant::now();
                let fidelity = run_solve(
                    &options.topology,
                    initial_mappings,
                    options.edge_to_edge,
                    options.beams,
                    options.paths,
                    options.qasm.clone(),
                    None,
                    false,
                )
                .fidelity;
                let elapsed = start.elapsed();
                (initial_mappings, fidelity, elapsed)
            })
            .for_each(|(initial_mappings, f, elapsed)| {
                println!("{}, {}, {}", initial_mappings, f, elapsed.as_nanos())
            });
    }

    /*
    const SHOTS: usize = 50;
    
    (3..100000).step_by(500)
        .map(|x| std::iter::repeat(x).take(SHOTS).collect::<Vec<_>>())
        .flatten()
        .collect::<Vec<_>>()
        .par_iter()
        .map(|&size| {
            let answer = beam_solve(&connection, &gates, RandomMapper::new(connection.size, size), 40);
            (size, answer.scheduler.fidelity)
        })
        .for_each(|(size, fidelity)| println!("{},{}", size, fidelity));
    */

    /*
    (1..100).step_by(5)
        .map(|x| std::iter::repeat(x).take(SHOTS))
        .flatten()
        .collect::<Vec<_>>()
        .par_iter()
        .map(|&beams| {
            let answer = beam_solve(&connection, &gates, RandomMapper::new(connection.size, 50000), beams);
            (beams, answer.scheduler.fidelity)
        })
        .for_each(|(beams, fidelity)| println!("{}, {}", beams, fidelity));
    */
    Ok(())
}
