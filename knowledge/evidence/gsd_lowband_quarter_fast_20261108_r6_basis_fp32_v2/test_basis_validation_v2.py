"""New explicit comparator version; actual regression and independent guard tests."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def accept():return load('basis_validator_v2',HERE/'accept_fast617.py')


def test_actual_32_object_regression_passes_v2_and_reports_original_failure():
    fixture=json.loads((HERE/'regression_32_terms.json').read_text())
    a=accept()
    result=a.basis_loss_comparison(fixture['terms'],fixture['raw_loss'],np.dtype('float32'))
    assert result['producer_replayed_sum']==8.811872482299805
    assert result['absolute_error']==0. and result['passed'] is True
    assert result['legacy_fp64_sum']==8.81187358757228
    assert result['legacy_absolute_error']==1.10527247443315e-6 and result['legacy_gate_passed'] is False
    assert result['tolerance']==1e-6 and result['validator_version']=='basis-fp32-v2'


def test_high_dynamic_range_uses_ordered_source_dtype_not_reverse_pairwise_or_mean():
    a=accept();terms=[1e8]+[1.]*31
    result=a.basis_loss_comparison(terms,100000000.,np.dtype('float32'))
    assert result['producer_replayed_sum']==100000000. and result['legacy_fp64_sum']==100000031.
    assert result['legacy_gate_passed'] is False
    for wrong in (100000032.,100000024.,100000031.,3125000.):
        with pytest.raises(AssertionError):a.basis_loss_comparison(terms,wrong,np.dtype('float32'))
    with pytest.raises(AssertionError):a.basis_loss_comparison(list(reversed(terms)),100000000.,np.dtype('float32'))


def test_float64_contract_stays_float64():
    result=accept().basis_loss_comparison([1e8]+[1.]*31,100000031.,np.dtype('float64'))
    assert result['producer_dtype']=='float64' and result['legacy_gate_passed'] is True and result['absolute_error']==0.


@pytest.mark.parametrize('count',[1,9,32,41])
def test_actual_batch_cardinality_includes_singleton_and_real_tails(count):
    result=accept().basis_loss_comparison([.25]*count,count*.25,np.dtype('float32'))
    assert result['per_object_count']==count and result['producer_replayed_sum']==count*.25


@pytest.mark.parametrize('terms,raw,dtype',[([1.],1.00001,'float32'),([float('nan')],1.,'float32'),([float('inf')],1.,'float32'),([1.],float('nan'),'float32'),([1.],float('inf'),'float32'),([1.],1.,'float16')])
def test_altered_raw_nonfinite_or_unsupported_dtype_rejects(terms,raw,dtype):
    with pytest.raises(AssertionError):accept().basis_loss_comparison(terms,raw,np.dtype(dtype))


def fixture(tmp_path):
    builder=load('local_fixture_builder',HERE/'basis_test_fixtures.py')
    path,contents=builder.sealed_run(tmp_path,count=1,rank=100)
    a=accept()
    for name,raw in list(contents.items()):
        if name.endswith('/diagnostics.json'):
            d=json.loads(raw)
            with np.load(__import__('io').BytesIO(contents[name.rsplit('/',1)[0]+'/decoded_first_checkpoint.npz']),allow_pickle=False) as z:first=z['decoded_xyz']
            d['trajectory']['probes']=[dict(decoded_geometry_sha256=a.tensor_identity(first)) for _ in range(3)]
            contents[name]=json.dumps(d).encode()
    return builder,a,builder.seal(path,contents),contents


def test_wrong_first_witness_rejects_even_when_basis_energy_is_identical(tmp_path):
    builder,a,path,contents=fixture(tmp_path)
    assert a.load_run(path,tmp_path)['projections']
    key='batches/gaussian/batch0000/decoded_first_checkpoint.npz'
    first=np.zeros((1,2048,3),dtype=np.float32);first[0,2047,0]=.5 # Outside selected basis, so energy remains zero.
    contents[key]=builder.npz(decoded_xyz=first)
    with pytest.raises(AssertionError):a.load_run(builder.seal(tmp_path/'wrong_first.zip',contents),tmp_path)


@pytest.mark.parametrize('kind',['basis','frame','raw','nonfinite'])
def test_existing_physical_and_nonfinite_guards_still_reject(tmp_path,kind):
    builder,a,path,contents=fixture(tmp_path)
    assert a.load_run(path,tmp_path)['projections']
    prefix='batches/gaussian/batch0000/'
    if kind=='basis':
        u=np.eye(2048,100,dtype=np.float32);u[0,0]=.5
        contents[prefix+'physical_bases.npz']=builder.npz(basis_0=u)
    elif kind=='frame':
        changed=np.ones((1,2048,3),dtype=np.float32)
        contents[prefix+'reference.npz']=builder.npz(prepared_x=changed,initial_h0_xyz=np.zeros_like(changed))
    elif kind=='raw':
        d=json.loads(contents[prefix+'diagnostics.json']);d['trajectory']['decoded_spectral']['records'][0]['raw_loss']=.01
        contents[prefix+'diagnostics.json']=json.dumps(d).encode()
    else:
        first=np.zeros((1,2048,3),dtype=np.float32);first[0,0,0]=np.nan
        contents[prefix+'decoded_first_checkpoint.npz']=builder.npz(decoded_xyz=first)
    with pytest.raises(AssertionError):a.load_run(builder.seal(tmp_path/'invalid.zip',contents),tmp_path)


def test_changed_basis_with_resealed_hash_still_fails_energy_gate(tmp_path):
    builder,a,path,contents=fixture(tmp_path)
    prefix='batches/gaussian/batch0000/'
    first=np.zeros((1,2048,3),dtype=np.float32);first[0,0,0]=1.
    contents[prefix+'decoded_first_checkpoint.npz']=builder.npz(decoded_xyz=first)
    d=json.loads(contents[prefix+'diagnostics.json'])
    d['trajectory']['probes'][0]['decoded_geometry_sha256']=a.tensor_identity(first)
    d['trajectory']['decoded_spectral']['records'][0]['raw_loss']=float(np.float32(1./6144.))
    contents[prefix+'diagnostics.json']=json.dumps(d).encode()
    result=a.load_run(builder.seal(tmp_path/'valid_nonzero.zip',contents),tmp_path)
    details=a.receipt_details(result)
    assert details['basis_validator_version']=='basis-fp32-v2'
    assert details['basis_validator_source']==a.identity(HERE/'accept_fast617.py')
    assert len(details['basis_projections'])==15 and all(row['schema_version']==2 for row in details['basis_projections'])
    u=np.eye(2048,100,dtype=np.float32);u[0,0]=2.
    contents[prefix+'physical_bases.npz']=builder.npz(basis_0=u)
    g=json.loads(contents[prefix+'graph.json']);g['basis_sha256'][0]=a.tensor_identity(u)
    contents[prefix+'graph.json']=json.dumps(g).encode()
    with pytest.raises(AssertionError):a.load_run(builder.seal(tmp_path/'wrong_energy.zip',contents),tmp_path)


def test_legacy_float64_diagnostic_replays_v1_order_not_python_builtin_sum():
    result=accept().basis_loss_comparison([1e16]+[1.]*31,1e16,np.dtype('float64'))
    assert result['legacy_fp64_sum']==1e16 and result['legacy_gate_passed'] is True


def test_basis_dtype_must_match_actual_producer_reference_even_with_valid_hash(tmp_path):
    builder,a,path,contents=fixture(tmp_path)
    prefix='batches/gaussian/batch0000/'
    u=np.eye(2048,100,dtype=np.float64)
    contents[prefix+'physical_bases.npz']=builder.npz(basis_0=u)
    g=json.loads(contents[prefix+'graph.json']);g['basis_sha256'][0]=a.tensor_identity(u)
    contents[prefix+'graph.json']=json.dumps(g).encode()
    with pytest.raises(AssertionError):a.load_run(builder.seal(tmp_path/'wrong_dtype.zip',contents),tmp_path)
