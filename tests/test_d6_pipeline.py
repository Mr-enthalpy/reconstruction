import numpy as np
import pytest
torch = pytest.importorskip('torch')
from experiments.d6.pipeline import run_pipeline


class FrozenN1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, y):
        self.seen = tuple(y.shape)
        return dict(spectral_cube_hat=y.repeat(1,21,1,1)*2,
                    reprojected_measurement=y)


def run(boxes, **kwargs):
    model=FrozenN1().eval().requires_grad_(False)
    y=torch.arange(48,dtype=torch.float32).reshape(1,1,6,8)-20
    out=run_pipeline(y,model,boxes,observation_id='synthetic',
                     model_sha256='test-fixture',candidate_source='fixed test boxes',**kwargs)
    return out,model


def test_full_field_independent_branches_and_aligned_features():
    out,model=run([[2,1,3,2]], classifier=lambda f: f[:,:2],classifier_id='wiring-fixture')
    assert model.seen==(1,1,6,8)
    assert out['proxy'].min()<0  # No hidden clipping or per-observation scaling.
    np.testing.assert_array_equal(out['cube'][:,0],out['proxy'][:,0]*2)
    assert out['features'][0,3]==2*out['features'][0,0]
    assert out['scores'].shape==(1,2)
    assert out['metadata']['scope']=='simulation-only'


@pytest.mark.parametrize('boxes',[[[-1,0,2,2]],[[7,0,2,2]],[[0,0,0,2]],[[.5,0,1,1]]])
def test_reject_invalid_coordinates(boxes):
    with pytest.raises(ValueError):run(boxes)


def test_zero_candidates_preserves_full_reconstruction():
    out,model=run(np.empty((0,4)))
    assert model.seen==(1,1,6,8) and out['features'].shape==(0,24)
    assert out['metadata']['classifier_status']=='not_run'


def test_reject_trainable_n1():
    with pytest.raises(ValueError,match='frozen'):
        run_pipeline(torch.zeros(1,1,6,8),FrozenN1(),[[0,0,1,1]],
                     observation_id='test',model_sha256='test',candidate_source='test')
