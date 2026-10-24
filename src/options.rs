//! UCI options, which `setoption` sets and `uci` lists.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `ucioption.h` and `ucioption.cpp`: Stockfish, a UCI chess playing engine
//! derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish developers
//! (see AUTHORS), under the GNU General Public License, version 3 or later.

use std::fmt;

use crate::misc::IStream;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OptionType {
    Check,
    Spin { min: i32, max: i32 },
    Combo,
    Button,
    String,
}

#[derive(Clone, Debug)]
pub struct UciOption {
    kind: OptionType,
    default_value: String,
    current_value: String,
}

impl UciOption {
    pub fn check(default: bool) -> UciOption {
        UciOption::with(OptionType::Check, if default { "true" } else { "false" })
    }

    pub fn spin(default: i32, min: i32, max: i32) -> UciOption {
        UciOption::with(OptionType::Spin { min, max }, &default.to_string())
    }

    /// A choice among `values`, as `uci` lists them after the default:
    /// "var" and each value.
    pub fn combo(default: &str, values: &[&str]) -> UciOption {
        let mut option = UciOption::with(OptionType::Combo, default);
        for value in values {
            option.default_value += " var ";
            option.default_value += value;
        }
        option
    }

    pub fn button() -> UciOption {
        UciOption::with(OptionType::Button, "")
    }

    pub fn string(default: &str) -> UciOption {
        UciOption::with(OptionType::String, default)
    }

    fn with(kind: OptionType, default: &str) -> UciOption {
        UciOption {
            kind,
            default_value: default.to_string(),
            current_value: default.to_string(),
        }
    }

    pub fn as_bool(&self) -> bool {
        debug_assert!(self.kind == OptionType::Check);
        self.current_value == "true"
    }

    pub fn as_int(&self) -> i32 {
        debug_assert!(matches!(self.kind, OptionType::Spin { .. }));
        self.current_value.parse().unwrap_or(0)
    }

    /// The value of a string, or a combo's choice as `setoption` gave it,
    /// in any case.
    pub fn as_str(&self) -> &str {
        debug_assert!(matches!(self.kind, OptionType::String | OptionType::Combo));
        &self.current_value
    }

    /// Sets the value as `setoption` gives it, unless it is invalid for the
    /// option, and returns whether it was set. GUIs should respect the limits,
    /// but users typing commands may not.
    pub fn set(&mut self, v: &str) -> bool {
        let valid = match self.kind {
            OptionType::Check => v == "true" || v == "false",
            OptionType::Spin { min, max } => value_in_range(v, min, max),
            OptionType::Combo => self
                .default_value
                .split(' ')
                .any(|value| value != "var" && value.eq_ignore_ascii_case(v)),
            OptionType::Button | OptionType::String => true,
        };
        if !valid {
            return false;
        }
        match self.kind {
            OptionType::String => {
                self.current_value = if v == "<empty>" {
                    String::new()
                } else {
                    v.to_string()
                }
            }
            OptionType::Button => {}
            _ => self.current_value = v.to_string(),
        }
        true
    }
}

/// Whether `v` is all a decimal integer, as C's strtoll reads it, from `min`
/// to `max`.
fn value_in_range(v: &str, min: i32, max: i32) -> bool {
    let digits = v.strip_prefix(['+', '-']).unwrap_or(v);
    if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit()) {
        return false;
    }
    v.parse::<i64>()
        .is_ok_and(|n| (min as i64..=max as i64).contains(&n))
}

/// The options in the order they are listed, found by names in any case.
#[derive(Clone, Debug, Default)]
pub struct OptionsMap {
    options: Vec<(String, UciOption)>,
}

impl OptionsMap {
    pub fn add(&mut self, name: &str, option: UciOption) {
        assert!(
            self.find(name).is_none(),
            "option \"{name}\" was already added"
        );
        self.options.push((name.to_string(), option));
    }

    fn find(&self, name: &str) -> Option<usize> {
        self.options
            .iter()
            .position(|(n, _)| n.eq_ignore_ascii_case(name))
    }

    /// The option named `name`, which must exist.
    pub fn get(&self, name: &str) -> &UciOption {
        match self.find(name) {
            Some(i) => &self.options[i].1,
            None => panic!("no option named \"{name}\""),
        }
    }

    /// Applies `setoption name NAME [value VALUE]` read from `is`, after
    /// "setoption": Ok with the option's name if the value was valid and set,
    /// for the engine to act on the change, Ok(None) if it was not, or Err
    /// with the message for an unknown option.
    pub fn setoption(&mut self, is: &mut IStream) -> Result<Option<&str>, String> {
        is.token(); // "name"

        // The name and the value can contain spaces.
        let mut name = String::new();
        while let Some(token) = is.token() {
            if token == "value" {
                break;
            }
            if !name.is_empty() {
                name.push(' ');
            }
            name += token;
        }

        let mut value = String::new();
        while let Some(token) = is.token() {
            if !value.is_empty() {
                value.push(' ');
            }
            value += token;
        }

        match self.find(&name) {
            Some(i) => {
                let (name, option) = &mut self.options[i];
                Ok(option.set(&value).then_some(name.as_str()))
            }
            None => Err(format!("No such option: {name}")),
        }
    }
}

/// The options as `uci` lists them, each after a newline.
impl fmt::Display for OptionsMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (name, o) in &self.options {
            let kind = match o.kind {
                OptionType::Check => "check",
                OptionType::Spin { .. } => "spin",
                OptionType::Combo => "combo",
                OptionType::Button => "button",
                OptionType::String => "string",
            };
            write!(f, "\noption name {name} type {kind}")?;
            match o.kind {
                OptionType::Check | OptionType::Combo => write!(f, " default {}", o.default_value)?,
                OptionType::Spin { min, max } => {
                    write!(f, " default {} min {min} max {max}", o.default_value)?
                }
                OptionType::String => {
                    let default = if o.default_value.is_empty() {
                        "<empty>"
                    } else {
                        &o.default_value
                    };
                    write!(f, " default {default}")?
                }
                OptionType::Button => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setoption_finds_names_in_any_case_and_checks_values() {
        let mut options = OptionsMap::default();
        options.add("UCI_Chess960", UciOption::check(false));
        options.add("Hash", UciOption::spin(16, 1, 1024));

        let mut is = IStream::new("name uci_chess960 value true");
        assert_eq!(options.setoption(&mut is), Ok(Some("UCI_Chess960")));
        assert!(options.get("UCI_Chess960").as_bool());

        let mut is = IStream::new("name UCI_Chess960 value TRUE");
        assert_eq!(options.setoption(&mut is), Ok(None));
        assert!(options.get("UCI_Chess960").as_bool());

        for (value, valid) in [
            ("64", true),
            ("+32", true),
            ("0", false),
            ("2048", false),
            ("1x", false),
        ] {
            let command = format!("name Hash value {value}");
            let mut is = IStream::new(&command);
            assert_eq!(
                options.setoption(&mut is),
                Ok(valid.then_some("Hash")),
                "{value}"
            );
        }
        assert_eq!(options.get("hash").as_int(), 32);

        let mut is = IStream::new("name Threads value 4");
        assert_eq!(
            options.setoption(&mut is),
            Err("No such option: Threads".to_string())
        );
    }

    #[test]
    fn combos_take_one_of_their_values() {
        let mut options = OptionsMap::default();
        options.add("Use NNUE", UciOption::combo("true", &["true", "pure"]));
        assert_eq!(
            options.to_string(),
            "\noption name Use NNUE type combo default true var true var pure"
        );
        assert_eq!(options.get("Use NNUE").as_str(), "true");

        for (value, valid) in [
            ("Pure", true),
            ("false", false),
            ("var", false),
            ("", false),
            ("maybe", false),
        ] {
            let command = format!("name Use NNUE value {value}");
            let mut is = IStream::new(&command);
            assert_eq!(
                options.setoption(&mut is),
                Ok(valid.then_some("Use NNUE")),
                "{value}"
            );
        }
        assert_eq!(options.get("Use NNUE").as_str(), "Pure");
    }

    #[test]
    fn options_are_listed_in_order() {
        let mut options = OptionsMap::default();
        options.add("Threads", UciOption::spin(1, 1, 1024));
        options.add("Clear Hash", UciOption::button());
        options.add("SyzygyPath", UciOption::string(""));
        options.add("UCI_Chess960", UciOption::check(false));
        assert_eq!(
            options.to_string(),
            "\noption name Threads type spin default 1 min 1 max 1024\
             \noption name Clear Hash type button\
             \noption name SyzygyPath type string default <empty>\
             \noption name UCI_Chess960 type check default false"
        );
    }
}
