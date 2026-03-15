pub mod error;
pub use error::MohuError;

pub type MohuResult<T> = std::result::Result<T, MohuError>;
