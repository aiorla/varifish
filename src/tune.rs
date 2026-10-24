//! Search constants that SPSA tunes, as UCI spin options of builds with the
//! `tune` feature.
//!
//! A tuning branch declares the constants it tunes in `tunables!` below,
//! with their defaults and ranges, and reads each where the search used it
//! with `tune!(NAME)`. Without the feature, `tune!(NAME)` is the constant
//! itself, so the search compiles as before; with it, each constant is an
//! option that `setoption` changes between searches. Tuned values land as
//! plain constants, and the branch's declarations stay off main.

#[cfg(feature = "tune")]
use std::sync::atomic::{AtomicI32, Ordering};

/// A tuned constant: its option's name, default, range and value.
#[cfg(feature = "tune")]
pub struct Tunable {
    pub name: &'static str,
    pub default: i32,
    pub min: i32,
    pub max: i32,
    value: AtomicI32,
}

#[cfg(feature = "tune")]
impl Tunable {
    pub const fn new(name: &'static str, default: i32, min: i32, max: i32) -> Tunable {
        Tunable {
            name,
            default,
            min,
            max,
            value: AtomicI32::new(default),
        }
    }

    #[inline(always)]
    pub fn get(&self) -> i32 {
        self.value.load(Ordering::Relaxed)
    }

    pub fn set(&self, value: i32) {
        self.value.store(value, Ordering::Relaxed);
    }
}

/// Declares the tuned constants, each `NAME = default, min, max;`: a
/// `Tunable` with the `tune` feature and a constant without it, and with the
/// feature `ALL`, the tunables in the order `uci` lists them.
macro_rules! tunables {
    ($($name:ident = $default:expr, $min:expr, $max:expr;)*) => {
        $(
            #[cfg(feature = "tune")]
            pub static $name: Tunable = Tunable::new(stringify!($name), $default, $min, $max);
            #[cfg(not(feature = "tune"))]
            pub const $name: i32 = $default;
        )*

        #[cfg(feature = "tune")]
        pub static ALL: &[&Tunable] = &[$(&$name),*];
    };
}

/// A tuned constant's value: its option's with the `tune` feature.
#[cfg(feature = "tune")]
#[macro_export]
macro_rules! tune {
    ($name:ident) => {
        $crate::tune::$name.get()
    };
}

/// A tuned constant's value: the constant without the `tune` feature.
#[cfg(not(feature = "tune"))]
#[macro_export]
macro_rules! tune {
    ($name:ident) => {
        $crate::tune::$name
    };
}

// The constants a tuning branch tunes: none on main.
tunables! {}

/// The tunable whose option is named `name`, in any case.
#[cfg(feature = "tune")]
pub fn find(name: &str) -> Option<&'static Tunable> {
    ALL.iter()
        .copied()
        .find(|tunable| tunable.name.eq_ignore_ascii_case(name))
}

#[cfg(test)]
mod tests {
    mod example {
        #[cfg(feature = "tune")]
        use crate::tune::Tunable;

        tunables! {
            FIRST = 7, 0, 10;
            SECOND = -3, -8, 8;
        }
    }

    #[cfg(not(feature = "tune"))]
    #[test]
    fn tunables_are_constants() {
        assert_eq!((example::FIRST, example::SECOND), (7, -3));
    }

    #[cfg(feature = "tune")]
    #[test]
    fn tunables_are_options() {
        let names: Vec<&str> = example::ALL.iter().map(|t| t.name).collect();
        assert_eq!(names, ["FIRST", "SECOND"]);
        assert_eq!((example::SECOND.min, example::SECOND.max), (-8, 8));
        assert_eq!(example::FIRST.get(), 7);
        example::FIRST.set(9);
        assert_eq!(example::FIRST.get(), 9);
    }
}
