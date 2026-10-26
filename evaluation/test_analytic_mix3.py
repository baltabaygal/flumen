"""Frozen mix3 production export: independent source goldens, API and N10."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import pytest
import flumen
from flumen.model import load_analytic_model

ROOT=Path(__file__).resolve().parents[1]
CKPT=ROOT/'model/checkpoints/analytic_mix3_v1'
GOLD=json.loads((Path(__file__).parent/'analytic_mix3_goldens.json').read_text())

@pytest.fixture(scope='module')
def model():return load_analytic_model()

def test_mix3_default_and_safe_checkpoint_manifest(model):
    assert model.version=='analytic' and model.seeds==tuple(range(8))
    assert model.guard=='current' and model.rise_tol==0.
    meta=json.loads((CKPT/'meta.json').read_text())
    assert meta['mixing_precision']=='source-float32' and len(meta['seeds'])==8
    assert 'Train-only' in meta['description']
    lines=(CKPT/'MANIFEST.sha256').read_text().splitlines()
    assert len(lines)==10
    for line in lines:
        h,name=line.split();assert hashlib.sha256((CKPT/name).read_bytes()).hexdigest()==h

@pytest.mark.parametrize('row',GOLD['pdf'])
def test_mix3_source_goldens(model,row):
    result=model.log_prob_lnmu(row['x'],row['params'])
    assert np.allclose(result,row['source_logpdf'],rtol=0,atol=2e-5)

@pytest.mark.parametrize('z',[.2,.8184,3.])
def test_mix3_planes_normalization_and_cutoff(model,z):
    p=[z,.674,.315,.811,.0493,.965,3402.,0.,-1.,0.];c=model.prepare(p)
    ua=math.log((16-c['m'])/c['s']-c['yb']+.2)
    x=c['m']+c['s']*(np.exp(np.linspace(math.log(1e-9),ua,80001))+c['yb']-.2)
    ps,pi=model.pdf_lnmu(x,p,'source'),model.pdf_lnmu(x,p,'image')
    assert np.trapezoid(ps,x)+.5*ps[-1]==pytest.approx(1.,abs=2e-3)
    assert np.trapezoid(pi,x)+pi[-1]==pytest.approx(1.,abs=2e-3)
    assert model.pdf_lnmu(c['cut']-1e-8,p)==0.
    assert np.allclose(model.pdf_mu(np.exp(x[::100]),p),ps[::100]/np.exp(x[::100]),rtol=1e-5,atol=1e-12)

@pytest.mark.parametrize('row',GOLD['n10'])
def test_mix3_sigma8_fisher_source_goldens(model,row):
    base=[.70,.25,.95,.045,.96,3400.,0.,-.85,-.2]
    a=np.linspace(-.12,.12,9);x=np.arange(-3.,9.,.00025)
    p=[]
    for aa in a:
        prm=[row['z'],*base];prm[3]=base[2]*np.exp(aa);p.append(model.pdf_lnmu(x,prm))
    p=np.array(p);nfft=1<<int(np.ceil(np.log2(2*len(x))))
    f=np.fft.rfftfreq(nfft,x[1]-x[0]);ft=np.fft.rfft(p,nfft,axis=1)
    weights=np.polynomial.polynomial.polyfit(a,np.eye(9),3)[1];n10=[]
    for e in GOLD['eps']:
        q=np.maximum(np.fft.irfft(ft*np.exp(-.5*(2*np.pi*f*2*e)**2),nfft,axis=1)[:,:len(x)],0)
        q/=np.trapezoid(q,x,axis=1)[:,None];mask=(q>1e-300).all(0)&(q[4]>1e-12)
        g=weights@np.log(q[:,mask]);n10.append(100/np.trapezoid(q[4,mask]*g*g,x[mask]))
    assert np.allclose(n10,row['source_N10'],rtol=2e-5,atol=0)

def test_mix3_public_api_low_z_without_extrapolation_warning(model):
    import warnings
    p=[.2,.674,.315,.811,.0493,.965,3402.,0.,-1.,0.]
    x=np.linspace(-.02,.08,200)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always');xx,pdf=flumen.generate_pdf_lnmu(z_s=.2,lnmu=x,backend='analytic')
    assert not any('outside the confirmed training range' in str(w.message) for w in recorded)
    assert np.array_equal(xx,x) and np.allclose(pdf,model.pdf_lnmu(x,p))
    assert flumen.ANALYTIC_TRAINING_RANGE['z_s']==(.2,12.)


def test_public_default_is_analytic(model):
    assert flumen.get_model().version == 'analytic'
    assert flumen.TRAINING_RANGE is flumen.ANALYTIC_TRAINING_RANGE
    x=np.linspace(-.08,.3,250)
    _,default=flumen.generate_pdf_lnmu(z_s=1.,lnmu=x)
    _,explicit=flumen.generate_pdf_lnmu(z_s=1.,lnmu=x,backend='analytic')
    assert np.array_equal(default,explicit)
    mu=np.exp(x)
    _,default_mu=flumen.generate_pdf(z_s=1.,mu=mu)
    assert np.allclose(default_mu,default/mu,rtol=1e-5,atol=1e-10)
    _,de=flumen.generate_pdf_lnmu(z_s=1.,lnmu=x,Ok=-.02,w0=-.9,wa=-.1)
    assert np.isfinite(de).all() and not np.allclose(default,de)
