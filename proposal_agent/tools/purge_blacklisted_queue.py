import os
import sys
import json
from pathlib import Path

# ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from proposal_agent.ui import calibration_dashboard as cd
from datetime import datetime
import getpass

QUEUE_PATH = cd.QUEUE_PATH
AUDIT_PATH = cd.AUDIT_PATH


def main(dry_run=True):
    cfg = cd.load_notifications_config()
    # detect blacklisted in current config
    blacklisted_found = False
    reasons = []
    for u in cfg.get('webhooks', []):
        try:
            host = __import__('urllib.parse').parse.urlparse(u).hostname
        except Exception:
            host = None
        if host:
            for bad in cd.DOMAIN_BLACKLIST:
                if host == bad or host.endswith('.' + bad):
                    blacklisted_found = True
                    reasons.append({'type': 'webhook', 'target': u, 'reason': f'host {host} blacklisted'})
    smtp = cfg.get('smtp') or {}
    server = smtp.get('server')
    if server:
        host = server.split(':')[0]
        for bad in cd.DOMAIN_BLACKLIST:
            if host == bad or host.endswith('.' + bad):
                blacklisted_found = True
                reasons.append({'type': 'smtp', 'target': server, 'reason': f'smtp host {host} blacklisted'})

    if not blacklisted_found:
        print('No blacklisted targets found in current config; nothing to purge.')
        return 0

    items = cd.read_notification_queue()
    count = len(items)
    if count == 0:
        print('Queue empty; nothing to purge.')
        return 0

    backup = None
    try:
        QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        backup = str(QUEUE_PATH.with_suffix('.jsonl.bak'))
        QUEUE_PATH.rename(QUEUE_PATH.with_suffix('.jsonl.bak'))
    except Exception:
        # fallback: copy contents
        try:
            txt = QUEUE_PATH.read_text(encoding='utf-8')
            backup = str(QUEUE_PATH.with_suffix('.jsonl.bak'))
            QUEUE_PATH.with_suffix('.jsonl.bak').write_text(txt, encoding='utf-8')
        except Exception:
            backup = None

    # purge: remove queue file
    try:
        if QUEUE_PATH.exists():
            QUEUE_PATH.unlink()
    except Exception:
        pass

    # write audit entry
    entry = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'actor': getpass.getuser(),
        'action': 'purge_blacklisted_queue',
        'count_removed': count,
        'backup': backup,
        'reasons': reasons
    }
    try:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_PATH.open('a', encoding='utf-8') as af:
            af.write(json.dumps(entry) + '\n')
    except Exception:
        pass

    print(f'Purged {count} queued notifications (backup: {backup})')
    return 0


if __name__ == '__main__':
    main(dry_run=False)
