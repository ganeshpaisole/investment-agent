import json
import sys
import os

# ensure workspace root is on sys.path so package imports work when running from terminal
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from proposal_agent.ui.calibration_dashboard import (
    load_notifications_config,
    validate_webhook,
    validate_smtp,
    read_notification_queue,
)

def main():
    out = {}
    cfg = load_notifications_config()
    out['config'] = cfg
    webhooks = cfg.get('webhooks', [])
    wh_results = []
    for u in webhooks:
        r = validate_webhook(u)
        wh_results.append(r)
    out['webhook_validation'] = wh_results
    smtp_cfg = cfg.get('smtp')
    out['smtp_validation'] = validate_smtp(smtp_cfg)
    queue = read_notification_queue()
    out['queue_count'] = len(queue)
    out['queue_preview'] = [{'timestamp': q.get('timestamp'), 'event': q.get('event'), 'attempts': q.get('attempts',0)} for q in queue[:20]]
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
