//! Flow module for RealNVP normalizing flows

mod coupling;
mod network;
mod actnorm;

pub use coupling::CouplingLayer;
pub use network::RealNVP;
pub use actnorm::ActNorm;
