#[cfg(any(test, feature = "quickcheck"))]
#[macro_use]
extern crate quickcheck;

#[cfg(feature = "quickcheck")]
quickcheck!{}

#[macro_use]
pub mod list;

pub mod treemap;

pub use list::List;
