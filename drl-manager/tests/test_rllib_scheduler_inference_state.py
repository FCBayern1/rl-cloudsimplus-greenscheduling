"""Regression for deployment dropout and memory lost by the baseline adapter."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.baselines.global_schedulers import RLlibNewAPIGlobalScheduler


class StatefulModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout=torch.nn.Dropout(.9)
        self.calls=[]

    def forward_inference(self,batch):
        self.calls.append(batch)
        h=batch.get('state_in',{}).get('memory',torch.zeros(1))
        h=h+1
        logits=self.dropout(torch.tensor([[[1.,2.,3.]]]))
        return {'action_dist_inputs':logits,'state_out':{'memory':h}}


def scheduler(stochastic=False):
    m=StatefulModule()
    algo=SimpleNamespace(env_runner=SimpleNamespace(module={'global_policy':m}))
    return RLlibNewAPIGlobalScheduler(3,1,algo,stochastic=stochastic),m


@pytest.mark.parametrize('stochastic',[False,True])
def test_network_eval_mode_is_independent_of_action_sampling(stochastic):
    s,m=scheduler(stochastic)
    assert not m.training and not m.dropout.training
    s.schedule({'x':np.zeros(1)})
    s.schedule({'x':np.zeros(1)})
    assert m.calls[1]['state_in']['memory'].item()==1
    assert s._recurrent_state['memory'].item()==2


def test_episode_reset_and_repeatability():
    s,m=scheduler()
    first=[s.schedule({'x':np.zeros(1)}) for _ in range(3)]
    s.reset()
    second=[s.schedule({'x':np.zeros(1)}) for _ in range(3)]
    assert first==second==[[2],[2],[2]]
    assert 'state_in' not in m.calls[0] and 'state_in' not in m.calls[3]
    assert s._recurrent_state['memory'].item()==3


def test_feedforward_module_remains_supported():
    class Stateless(torch.nn.Module):
        def forward_inference(self,batch):
            assert 'state_in' not in batch
            return {'action_dist_inputs':torch.tensor([[[1.,0.,0.]]])}
    m=Stateless();a=SimpleNamespace(env_runner=SimpleNamespace(module={'global_policy':m}))
    s=RLlibNewAPIGlobalScheduler(3,1,a)
    assert s.schedule({'x':np.zeros(1)})==s.schedule({'x':np.zeros(1)})==[0]
