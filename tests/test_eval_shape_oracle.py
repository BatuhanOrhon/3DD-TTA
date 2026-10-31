"""Small oracle-only scope and object-ID guards, without native imports."""
import copy
import hashlib
import importlib
import json

import numpy as np
import pytest

import test_eval_shape_latent as parent


def feature():
    return importlib.import_module('scripts.eval_shape_oracle')


def args(root, stage='development', count=64):
    return parent.args(root, stage, count)+['--oracle-state-root', str(root/'oracle_states')]


@pytest.mark.parametrize('arm', ['baseline','clean_z_fixed_h','clean_z_reencode_h','clean_pair'])
def test_fixed_oracle_scope_and_no_global_or_style_adapter(tmp_path, arm):
    runner = feature()
    cfg, = runner.configurations(args(tmp_path)+['--oracle-arm',arm])
    assert cfg['method'] == runner.METHOD
    assert cfg['shape_oracle']['arm'] == arm
    assert cfg['shape_oracle']['global_prior_calls'] == 0
    assert 'shape_latent' not in cfg
    assert cfg['decoder_style'] == 'original'
    assert 'oracle' in cfg['shared_reconstruction'].lower()
    assert 'same initial z/h' not in cfg['shared_reconstruction']
    runner.validate_config(cfg)
    for key,value in [('normal_reverse_steps',5),('seed',1),('decoder_style','updated'),('max_batches',1)]:
        with pytest.raises(ValueError):runner.validate_config(dict(cfg,**{key:value}))
    bad=copy.deepcopy(cfg);bad['shape_oracle']['global_prior_calls']=1
    with pytest.raises(ValueError):runner.validate_config(bad)


def test_only_explicit_32_smoke_or_64_all15_and_disjoint_cache_roots(tmp_path):
    runner=feature()
    cfg,=runner.configurations(args(tmp_path,'smoke',32)+['--oracle-arm','baseline','--oracle-technical-smoke'])
    assert cfg['corruptions']==['gaussian','background']
    assert cfg['technical_smoke']
    with pytest.raises(ValueError):runner.configurations(args(tmp_path,'development',32)+['--oracle-arm','baseline'])
    cfg['oracle_state_root']=cfg['paired_state_root']
    with pytest.raises(ValueError):runner.validate_config(cfg)


def test_batch_hash_must_map_to_exact_selected_original_ids(tmp_path):
    runner=feature();cfg,=runner.configurations(args(tmp_path)+['--oracle-arm','clean_pair'])
    ids=cfg['selected_original_ids'][32:64]
    digest=hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()
    assert runner.batch_ids(cfg,tmp_path/(digest+'.npz'))==ids
    with pytest.raises(ValueError):runner.batch_ids(cfg,tmp_path/'unknown.npz')
