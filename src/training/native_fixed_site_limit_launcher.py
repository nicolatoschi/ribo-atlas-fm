#!/usr/bin/env python3
"""Apply the verified site wall limit to collectives and record the runtime override.

The original frozen driver, model, tensors, RNG, numerical checks and optimizer
are unchanged. Report decoration happens after the timed update.
"""
import datetime
from pathlib import Path
import native_fixed_launcher_driver as native

SITE_LIMIT_SECONDS = 12 * 60 * 60


def main():
    path = Path(__file__).resolve()
    original_init = native.driver.dist.init_process_group
    original_write = native.driver.atomic_json
    manifest = __import__('json').loads(path.with_name('frozen_manifest.json').read_text())
    contract = {
        'wrapper_sha256': native.driver.sha(path),
        'original_freeze_sha256': manifest['freeze_sha256'],
        'site_max_wall_seconds': SITE_LIMIT_SECONDS,
        'collective_timeout_seconds': SITE_LIMIT_SECONDS,
        'scope': 'Runtime timeout override and after-update metadata only; numerical source and guards unchanged',
    }

    def site_init(*args, **kwargs):
        kwargs['timeout'] = datetime.timedelta(seconds=SITE_LIMIT_SECONDS)
        return original_init(*args, **kwargs)

    def site_write(destination, report):
        if 'rank' in report and 'global_constructs' in report:
            report['site_runtime_limit_contract'] = dict(contract)
        return original_write(destination, report)

    native.driver.dist.init_process_group = site_init
    native.driver.atomic_json = site_write
    native.main()


if __name__ == '__main__':
    main()
