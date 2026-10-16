"""Run the frozen batched driver unchanged; optionally force autograd anomaly mode off."""
import sys, time, torch
mode = sys.argv.pop(1)
assert mode in ('on', 'off')
if mode == 'off':
    original = torch.autograd.set_detect_anomaly
    torch.autograd.set_detect_anomaly = lambda *a, **k: original(False)
packet = sys.argv.pop(1)
sys.path.insert(0, packet)
import batched_full_trajectory_driver as driver
started = time.perf_counter()
driver.main()
print('anomaly_enabled_at_exit', torch.is_anomaly_enabled(), 'wall', round(time.perf_counter() - started, 1))
