//! The Makefile downloads each variant's network by the name the engine
//! loads by default, so that `make atomic-net`, `make 3check-net`, `make
//! crazyhouse-net` and `make antichess-net` fetch the networks the program
//! looks for, and checks each against a whole SHA-256 whose first 12 digits
//! are the name's.

use varifish::nnue::{antichess, atomic, crazyhouse, threecheck};

/// The value the Makefile gives `variable`, as `VARIABLE = value`.
fn makefile_value(makefile: &str, variable: &str) -> String {
    let prefix = format!("{variable} = ");
    makefile
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("the Makefile sets no {variable}"))
        .to_string()
}

#[test]
fn the_makefile_names_and_pins_each_variants_default_network() {
    let makefile = include_str!("../Makefile");
    for (variable, prefix, default) in [
        ("ATOMIC_NET", "atomic-", atomic::EVAL_FILE_DEFAULT_NAME),
        (
            "THREE_CHECK_NET",
            "3check-",
            threecheck::EVAL_FILE_DEFAULT_NAME,
        ),
        (
            "CRAZYHOUSE_NET",
            "crazyhouse-",
            crazyhouse::EVAL_FILE_DEFAULT_NAME,
        ),
        (
            "ANTICHESS_NET",
            "antichess-",
            antichess::EVAL_FILE_DEFAULT_NAME,
        ),
    ] {
        let name = makefile_value(makefile, variable);
        assert_eq!(name, default, "{variable}");
        let sha256 = makefile_value(makefile, &format!("$({variable})_SHA256"));
        assert_eq!(sha256.len(), 64, "{variable}'s SHA-256");
        assert!(
            sha256.bytes().all(|b| b.is_ascii_hexdigit()),
            "{variable}'s SHA-256"
        );
        let digest = name
            .strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix(".nnue"))
            .unwrap_or_else(|| panic!("{name} is not {prefix}<12 digits>.nnue"));
        assert_eq!(digest, &sha256[..12], "{name} against its SHA-256");
    }
}
