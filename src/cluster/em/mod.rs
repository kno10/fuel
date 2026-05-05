pub mod models;
mod optimizer;

pub use models::diagonal::{DiagonalGaussianModel, DiagonalGaussianModelFactory};
pub use models::multivariate::{MultivariateGaussianModel, MultivariateGaussianModelFactory};
pub use models::spherical::{SphericalGaussianModel, SphericalGaussianModelFactory};
pub use models::textbook_diagonal::{
    TextbookDiagonalGaussianModel, TextbookDiagonalGaussianModelFactory,
};
pub use models::textbook_multivariate::{
    TextbookMultivariateGaussianModel, TextbookMultivariateGaussianModelFactory,
};
pub use models::textbook_spherical::{
    TextbookSphericalGaussianModel, TextbookSphericalGaussianModelFactory,
};
pub use models::two_pass_diagonal::{
    TwoPassDiagonalGaussianModel, TwoPassDiagonalGaussianModelFactory,
};
pub use models::two_pass_multivariate::{
    TwoPassMultivariateGaussianModel, TwoPassMultivariateGaussianModelFactory,
};
pub use models::two_pass_spherical::{
    TwoPassSphericalGaussianModel, TwoPassSphericalGaussianModelFactory,
};
pub use models::von_mises_fisher::{VonMisesFisherModel, VonMisesFisherModelFactory};
pub use optimizer::{EmConfig, EmModel, EmResult, expectation_maximization, log_sum_exp};
