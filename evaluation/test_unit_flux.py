"""Independent moment integration, exact known distribution, and failure handling."""
import numpy as np
import pytest
from scipy.integrate import quad
from flumen.model import load_model
from flumen.model.flux import unit_flux_context, FluxCalibrationError


class GaussianBody:
    @staticmethod
    def controls(z, theta):
        return .8, .3, -12., -14.


def test_known_lognormal_shift_and_mass():
    def density(x,z,theta):
        return 3*np.exp(-.5*((x-.8)/.3)**2)/(.3*np.sqrt(2*np.pi))
    norm,delta=unit_flux_context(density,GaussianBody(),1.,())
    assert norm == pytest.approx(3.,abs=2e-6)
    assert delta == pytest.approx(-.8+.3**2/2,abs=2e-6)
    moment=quad(lambda x:np.exp(-x)*density(x-delta,1.,())/norm,-5,5,epsabs=1e-10)[0]
    assert moment == pytest.approx(1.,abs=2e-6)


def test_invalid_density_fails_instead_of_false_unit_claim():
    with pytest.raises(FluxCalibrationError):
        unit_flux_context(lambda x,z,t:np.full_like(x,np.nan),GaussianBody(),1.,())


def test_public_unit_flux_with_independent_uniform_grid():
    """At an ordinary context (well inside the FLUX_DELTA_MAX=0.05 trust
    region -- see composite.py's 2026-09-07 comment), the default 'unit'
    mode's shift is unclipped and unit flux holds to full numerical
    accuracy."""
    z,theta=2.,(.674,.315,.811,.0493,.965,3.402)
    model=load_model();x=np.linspace(-8,30,70001)
    p=model.pdf_lnmu(x,z,theta)
    assert np.trapezoid(p,x)==pytest.approx(1.,abs=5e-6)
    assert np.trapezoid(p*np.exp(-x),x)==pytest.approx(1.,abs=5e-6)
    assert model.flux_mode=='unit'


def test_extreme_context_is_capped_not_exact():
    """At an extreme (z, theta) needing a large shift, the DEFAULT policy
    caps |delta| at FLUX_DELTA_MAX rather than enforcing exact unit flux --
    a 2026-09-07 change from the first unit-flux implementation, made after
    the qualification run showed uncapped shifts at this class of context
    cost a ~5x median KL regression at z>=8 (correlation(|delta|,kl_unit)
    =0.974 across the 973-context test set; every worst-affected context
    had `legacy_delta` saturated at its own cap already). This context
    (z=10.43) is exactly the one the ORIGINAL version of this test used to
    assert EXACT unit flux for -- confirm it is now capped, and that the
    uncapped `flux.unit_flux_context` utility still honestly reports what
    exact unit flux would have required (large, as expected)."""
    import flumen.model.composite as comp
    from flumen.model.flux import unit_flux_context
    from flumen.model.hybrid_body import load_hybrid_body
    z,theta=10.43,(.576,.496,1.491,.0473,.9876,3.4401)

    model=load_model()
    model._log_prob.context.cache_clear()
    norm,delta=model._log_prob.context(z,theta)
    assert abs(delta)==pytest.approx(comp.FLUX_DELTA_MAX,abs=1e-9), (
        "expected the cap to bind at this known-extreme context")

    body=load_hybrid_body()
    _,uncapped_delta=unit_flux_context(model._log_prob.unnormalized,body,z,theta)
    assert abs(uncapped_delta)>0.5, (
        "the uncapped utility should still honestly report a large "
        "shift here -- if this fails, the context stopped being extreme "
        "and this test should be re-pointed at one that still is")

    x=np.linspace(-8,30,70001)
    p=model.pdf_lnmu(x,z,theta)
    inverse_mean=np.trapezoid(p*np.exp(-x),x)
    assert abs(inverse_mean-1.)>1e-4, (
        "capping means <1/mu>=1 is now DELIBERATELY not exact here -- if "
        "this assertion fails the cap silently stopped applying")
    assert np.trapezoid(p,x)==pytest.approx(1.,abs=5e-6), (
        "normalization itself must stay exact even when the shift is capped")


def test_invalid_mode_is_rejected():
    from flumen.model.composite import make_composite
    with pytest.raises(ValueError,match='flux_mode'):
        make_composite(None,flux_mode='typo')
