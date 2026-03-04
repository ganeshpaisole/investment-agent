"""
Simple, safe test script for notifications.
- Reads `proposal_agent/data/notifications.json`.
- If `enabled` is false (default), prints what would be sent and exits.
- If `enabled` is true, attempts webhook POSTs and SMTP send (careful: will contact network).

Usage:
python proposal_agent/tools/test_notifications.py

"""
import os
import json
import ssl
import smtplib
import urllib.request
from pathlib import Path

BASE = Path(__file__).parents[1]
NOTIFY_PATH = BASE / 'data' / 'notifications.json'

PAYLOAD = {
    'event': 'test_notification',
    'payload': {
        'message': 'This is a test notification from Proposal Agent',
    }
}


def load_config():
    if not NOTIFY_PATH.exists():
        print('notifications.json not found at', NOTIFY_PATH)
        return None
    try:
        return json.loads(NOTIFY_PATH.read_text(encoding='utf-8'))
    except Exception as e:
        print('Failed to read notifications.json:', e)
        return None


def overlay_env(cfg):
    if cfg is None:
        cfg = {}
    # env overrides
    if os.getenv('PROPOSAL_NOTIFY_ENABLED') is not None:
        cfg['enabled'] = os.getenv('PROPOSAL_NOTIFY_ENABLED').lower() in ('1', 'true', 'yes')
    if os.getenv('PROPOSAL_NOTIFY_WEBHOOKS'):
        cfg['webhooks'] = [u.strip() for u in os.getenv('PROPOSAL_NOTIFY_WEBHOOKS').split(',') if u.strip()]
    smtp = cfg.get('smtp', {}) or {}
    if os.getenv('PROPOSAL_NOTIFY_SMTP_SERVER'):
        smtp['server'] = os.getenv('PROPOSAL_NOTIFY_SMTP_SERVER')
    if os.getenv('PROPOSAL_NOTIFY_SMTP_PORT'):
        try:
            smtp['port'] = int(os.getenv('PROPOSAL_NOTIFY_SMTP_PORT'))
        except Exception:
            pass
    if os.getenv('PROPOSAL_NOTIFY_SMTP_FROM'):
        smtp['from'] = os.getenv('PROPOSAL_NOTIFY_SMTP_FROM')
    if os.getenv('PROPOSAL_NOTIFY_SMTP_TO'):
        smtp['to'] = [s.strip() for s in os.getenv('PROPOSAL_NOTIFY_SMTP_TO').split(',') if s.strip()]
    if os.getenv('PROPOSAL_NOTIFY_SMTP_USERNAME'):
        smtp['username'] = os.getenv('PROPOSAL_NOTIFY_SMTP_USERNAME')
    if os.getenv('PROPOSAL_NOTIFY_SMTP_PASSWORD'):
        smtp['password'] = os.getenv('PROPOSAL_NOTIFY_SMTP_PASSWORD')
    if os.getenv('PROPOSAL_NOTIFY_USE_TLS') is not None:
        smtp['use_tls'] = os.getenv('PROPOSAL_NOTIFY_USE_TLS').lower() in ('1', 'true', 'yes')
    if smtp:
        cfg['smtp'] = smtp
    return cfg


def dry_run_report(cfg):
    print('\nNotifications config (dry run):')
    print(json.dumps(cfg, indent=2))
    print('\nWould send webhooks to:')
    for u in cfg.get('webhooks', []):
        print(' -', u)
    smtp = cfg.get('smtp')
    if smtp:
        print('\nWould send email via:')
        print(' server:', smtp.get('server'))
        print(' port:', smtp.get('port'))
        print(' from:', smtp.get('from'))
        print(' to:', smtp.get('to'))
    print('\nTo actually send network notifications, set "enabled": true in the config (be careful).')


def send_webhooks(cfg):
    urls = cfg.get('webhooks', [])
    if not urls:
        print('No webhooks configured.')
        return
    data = json.dumps(PAYLOAD).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    for url in urls:
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                print('Webhook', url, '->', r.status)
        except Exception as e:
            print('Webhook', url, 'failed:', e)


def send_email(cfg):
    smtp = cfg.get('smtp')
    if not smtp:
        print('No SMTP configured.')
        return
    to_list = smtp.get('to', [])
    if not to_list:
        print('SMTP configured but no recipients.')
        return
    subject = 'Proposal Agent Test Notification'
    body = json.dumps(PAYLOAD, indent=2)
    msg = f"From: {smtp.get('from')}\r\nTo: {', '.join(to_list)}\r\nSubject: {subject}\r\n\r\n{body}"
    server = smtp.get('server', 'localhost')
    port = smtp.get('port', 25)
    use_tls = smtp.get('use_tls', False)
    username = smtp.get('username')
    password = smtp.get('password')
    try:
        if use_tls:
            context = ssl.create_default_context()
            with smtplib.SMTP(server, port, timeout=10) as s:
                s.starttls(context=context)
                if username and password:
                    s.login(username, password)
                s.sendmail(smtp.get('from'), to_list, msg)
                print('Email sent to', to_list)
        else:
            with smtplib.SMTP(server, port, timeout=10) as s:
                if username and password:
                    s.login(username, password)
                s.sendmail(smtp.get('from'), to_list, msg)
                print('Email sent to', to_list)
    except Exception as e:
        print('SMTP send failed:', e)


if __name__ == '__main__':
    cfg = load_config()
    if not cfg:
        raise SystemExit(1)
    cfg = overlay_env(cfg)
    if not cfg.get('enabled'):
        dry_run_report(cfg)
        raise SystemExit(0)
    print('Sending webhooks...')
    send_webhooks(cfg)
    print('Sending emails...')
    send_email(cfg)
    print('Done')
