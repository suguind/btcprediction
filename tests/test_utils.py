import sys
import types
import numpy as np
import datetime

# Stub torch module and other dependencies to satisfy imports
if 'torch' not in sys.modules:
    torch_stub = types.ModuleType('torch')
    torch_stub.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda seed: None)
    torch_stub.manual_seed = lambda seed: None
    torch_stub.device = lambda *args, **kwargs: 'cpu'
    sys.modules['torch'] = torch_stub

if 'prometheus_client' not in sys.modules:
    prom_client = types.ModuleType('prometheus_client')
    prom_client.Gauge = lambda *args, **kwargs: None
    sys.modules['prometheus_client'] = prom_client

if 'pytz' not in sys.modules:
    pytz_stub = types.ModuleType('pytz')
    class UTC(datetime.tzinfo):
        def utcoffset(self, dt):
            return datetime.timedelta(0)
        def dst(self, dt):
            return datetime.timedelta(0)
    pytz_stub.UTC = UTC()
    sys.modules['pytz'] = pytz_stub

import btcprediction.utils as utils


def test_advanced_time_warp_shape():
    x = np.random.rand(20, 5)
    warped = utils.advanced_time_warp(x, num_control_points=4, sigma=0.3)
    assert warped.shape == x.shape


def test_advanced_time_warp_sigma_zero():
    x = np.random.rand(15, 3)
    warped = utils.advanced_time_warp(x, num_control_points=4, sigma=0)
    np.testing.assert_allclose(warped, x)
