import streamlit as st
import pandas as pd
import json
from pathlib import Path
from statistics import median
import getpass
from datetime import datetime
import os
import smtplib
import ssl
import json as _json
import urllib.request
import urllib.error
import urllib.parse
import time
import socket
import sys


AUDIT_PATH = Path(__file__).parents[1] / 'data' / 'calibration_audit.jsonl'
PENDING_PATH = Path(__file__).parents[1] / 'data' / 'calibration_pending.jsonl'
APPROVERS_PATH = Path(__file__).parents[1] / 'data' / 'approvers.json'

MODEL_PATH = Path(__file__).parents[1] / 'data' / 'estimation_model.json'
HIST_PATH = Path(__file__).parents[1] / 'data' / 'historical_bids_example.csv'
CUSTOMER_ADJ = Path(__file__).parents[1] / 'data' / 'customer_adjustments.json'
NOTIFY_PATH = Path(__file__).parents[1] / 'data' / 'notifications.json'
QUEUE_PATH = Path(__file__).parents[1] / 'data' / 'notification_queue.jsonl'
# Domains to never send notifications to (placeholder/example/test domains)
DOMAIN_BLACKLIST = [
    'example.com',
    'example.org',
    'example.net',
    'localhost'
]


def load_rows(path):
    return pd.read_csv(path)


def compute_median_ratio(df, product=None):
    d = df.copy()
    d = d[d['estimated_total'] > 0]
    if product:
        d = d[d['product'].str.lower() == product.lower()]
    if d.empty:
        return None
    ratios = (d['actual_total'] / d['estimated_total']).dropna()
    if ratios.empty:
        return None
    return float(median(ratios))


def load_notifications_config():
    cfg = {}
    if NOTIFY_PATH.exists():
        try:
            cfg = json.loads(NOTIFY_PATH.read_text(encoding='utf-8'))
        except Exception:
            cfg = {}

    # Environment variable overrides for secrets and quick setup
    # PROPOSAL_NOTIFY_ENABLED (true/false)
    if os.getenv('PROPOSAL_NOTIFY_ENABLED') is not None:
        cfg['enabled'] = os.getenv('PROPOSAL_NOTIFY_ENABLED').lower() in ('1', 'true', 'yes')
    # PROPOSAL_NOTIFY_WEBHOOKS (comma-separated URLs)
    if os.getenv('PROPOSAL_NOTIFY_WEBHOOKS'):
        cfg['webhooks'] = [u.strip() for u in os.getenv('PROPOSAL_NOTIFY_WEBHOOKS').split(',') if u.strip()]

    smtp = cfg.get('smtp', {}) or {}
    if os.getenv('PROPOSAL_NOTIFY_SMTP_SERVER'):
        smtp['server'] = os.getenv('PROPOSAL_NOTIFY_SMTP_SERVER')
    if os.getenv('PROPOSAL_NOTIFY_SMTP_PORT'):
        try:
            smtp['port'] = int(os.getenv('PROPOSAL_NOTIFY_SMTP_PORT'))
        except Exception:
            smtp['port'] = cfg.get('smtp', {}).get('port', 25)
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


def send_webhook(event, payload):
    cfg = load_notifications_config()
    urls = cfg.get('webhooks', [])
    if not urls:
        return
    # Block webhook URLs that use placeholder/example domains
    for url in urls:
        try:
            parsed = urllib.parse.urlparse(url)
            host = parsed.hostname or ''
            for bad in DOMAIN_BLACKLIST:
                if host == bad or host.endswith('.' + bad):
                    entry = {
                        'timestamp': datetime.utcnow().isoformat() + 'Z',
                        'actor': getpass.getuser(),
                        'action': 'notification_blocked_blacklist',
                        'event': event,
                        'target': url,
                        'reason': f'host {host} is blacklisted'
                    }
                    try:
                        with AUDIT_PATH.open('a', encoding='utf-8') as af:
                            af.write(json.dumps(entry) + '\n')
                    except Exception:
                        pass
                    raise Exception(f'Webhook target blocked (blacklisted host): {host}')
        except Exception:
            continue
    body = {'event': event, 'payload': payload}
    data = json.dumps(body).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    errors = []
    # retry settings
    max_attempts = 3
    backoff = [1, 2, 4]
    for url in urls:
        last_exc = None
        for attempt in range(1, max_attempts + 1):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method='POST')
                with urllib.request.urlopen(req, timeout=10) as resp:
                    resp.read()
                last_exc = None
                break
            except urllib.error.HTTPError as he:
                # For 405, attempt OPTIONS to inspect allowed methods
                try:
                    allowed = None
                    opt = urllib.request.Request(url, method='OPTIONS')
                    with urllib.request.urlopen(opt, timeout=5) as resp:
                        allowed = resp.getheader('Allow')
                except Exception:
                    allowed = None
                try:
                    body = he.read().decode('utf-8', errors='ignore')
                except Exception:
                    body = ''
                msg = f'HTTPError {he.code} for {url}: {he.reason}; allow: {allowed}; body: {body}'
                last_exc = msg
                # 4xx are client errors -> no retry except 429
                if 400 <= he.code < 500 and he.code != 429:
                    break
            except urllib.error.URLError as ue:
                last_exc = f'URLError for {url}: {ue.reason}'
            except Exception as e:
                last_exc = f'Unexpected error for {url}: {type(e).__name__}: {e}'

            # backoff before next attempt
            if attempt < max_attempts:
                time.sleep(backoff[min(attempt - 1, len(backoff) - 1)])

        if last_exc:
            errors.append(last_exc)

    if errors:
        # write to audit for diagnostics
        entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'actor': getpass.getuser(),
            'action': 'notification_webhook_error',
            'event': event,
            'errors': errors
        }
        try:
            with AUDIT_PATH.open('a', encoding='utf-8') as af:
                af.write(json.dumps(entry) + '\n')
        except Exception:
            pass
        raise Exception('; '.join(errors))


def send_email(event, payload):
    cfg = load_notifications_config()
    smtp_cfg = cfg.get('smtp')
    if not smtp_cfg:
        return
    subject = f"Proposal Agent Notification: {event}"
    body = json.dumps(payload, indent=2)
    to_list = smtp_cfg.get('to', [])
    if not to_list:
        return
    msg = f"From: {smtp_cfg.get('from')}\r\nTo: {', '.join(to_list)}\r\nSubject: {subject}\r\n\r\n{body}"
    server = smtp_cfg.get('server', 'localhost')
    port = smtp_cfg.get('port', 25)
    # Block sending to blacklisted SMTP hosts
    try:
        host = server.split(':')[0]
        for bad in DOMAIN_BLACKLIST:
            if host == bad or host.endswith('.' + bad):
                entry = {
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'actor': getpass.getuser(),
                    'action': 'notification_blocked_blacklist',
                    'event': event,
                    'target': server,
                    'reason': f'smtp server {host} is blacklisted'
                }
                try:
                    with AUDIT_PATH.open('a', encoding='utf-8') as af:
                        af.write(json.dumps(entry) + '\n')
                except Exception:
                    pass
                raise Exception(f'SMTP server blocked (blacklisted host): {host}')
    except Exception:
        # fall through; validation functions will catch issues
        pass
    use_tls = smtp_cfg.get('use_tls', False)
    username = smtp_cfg.get('username')
    password = smtp_cfg.get('password')

    # retry logic for SMTP
    max_attempts = 3
    backoff = [1, 2, 4]
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            if use_tls:
                context = ssl.create_default_context()
                with smtplib.SMTP(server, port, timeout=10) as s:
                    s.starttls(context=context)
                    if username and password:
                        s.login(username, password)
                    s.sendmail(smtp_cfg.get('from'), to_list, msg)
            else:
                with smtplib.SMTP(server, port, timeout=10) as s:
                    if username and password:
                        s.login(username, password)
                    s.sendmail(smtp_cfg.get('from'), to_list, msg)
            last_err = None
            break
        except Exception:
            exc = sys.exc_info()[1]
            if isinstance(exc, socket.gaierror):
                last_err = f'DNS resolution failed for SMTP server {server}: {exc}'
            else:
                last_err = f'SMTP send failed: {exc}'
            if attempt < max_attempts:
                time.sleep(backoff[min(attempt - 1, len(backoff) - 1)])

    if last_err:
        # audit
        entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'actor': getpass.getuser(),
            'action': 'notification_smtp_error',
            'event': event,
            'error': str(last_err)
        }
        try:
            with AUDIT_PATH.open('a', encoding='utf-8') as af:
                af.write(json.dumps(entry) + '\n')
        except Exception:
            pass
        raise Exception(last_err)


def send_notification(event, payload):
    errors = []
    try:
        send_webhook(event, payload)
    except Exception as e:
        errors.append(f'webhook:{e}')
    try:
        send_email(event, payload)
    except Exception as e:
        errors.append(f'email:{e}')
    errors = []
    try:
        send_webhook(event, payload)
    except Exception as e:
        errors.append(f'webhook:{e}')
    try:
        send_email(event, payload)
    except Exception as e:
        errors.append(f'email:{e}')

    if errors:
        # write consolidated audit
        entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'actor': getpass.getuser(),
            'action': 'notification_send_failed',
            'event': event,
            'errors': errors
        }
        try:
            with AUDIT_PATH.open('a', encoding='utf-8') as af:
                af.write(json.dumps(entry) + '\n')
        except Exception:
            pass
        # Persist failed notification to queue for later retry
        try:
            QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
            qent = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'event': event,
                'payload': payload,
                'errors': errors,
                'attempts': 0
            }
            with QUEUE_PATH.open('a', encoding='utf-8') as qf:
                qf.write(json.dumps(qent) + '\n')
        except Exception:
            pass
        raise Exception('Notification errors: ' + '; '.join(errors))


def enqueue_failed_notification(event, payload, errors):
    try:
        QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        qent = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event': event,
            'payload': payload,
            'errors': errors,
            'attempts': 0
        }
        with QUEUE_PATH.open('a', encoding='utf-8') as qf:
            qf.write(json.dumps(qent) + '\n')
    except Exception:
        pass


def read_notification_queue():
    items = []
    if not QUEUE_PATH.exists():
        return items
    try:
        lines = [l for l in QUEUE_PATH.read_text(encoding='utf-8').splitlines() if l.strip()]
        for l in lines:
            try:
                items.append(json.loads(l))
            except Exception:
                continue
    except Exception:
        return []
    return items


def process_notification_queue(sender_callable, max_attempts=3):
    """Process queued notification entries using the provided sender_callable(event,payload).
    Returns a dict with counts: {processed: int, succeeded: int, failed: int, errors: [...]}
    """
    items = read_notification_queue()
    if not items:
        return {'processed': 0, 'succeeded': 0, 'failed': 0, 'errors': []}
    remaining = []
    succeeded = 0
    errors = []
    for it in items:
        attempts = it.get('attempts', 0) or 0
        if attempts >= max_attempts:
            # give up after max attempts; keep in audit but drop from queue
            errors.append({'entry': it, 'error': 'max_attempts_exceeded'})
            continue
        try:
            sender_callable(it.get('event'), it.get('payload'))
            succeeded += 1
        except Exception as e:
            # increment attempts and requeue
            it['attempts'] = attempts + 1
            it.setdefault('errors', []).append(str(e))
            remaining.append(it)
            errors.append({'entry': it, 'error': str(e)})

    # overwrite queue with remaining items
    try:
        if remaining:
            QUEUE_PATH.write_text('\n'.join([json.dumps(r) for r in remaining]), encoding='utf-8')
        else:
            try:
                QUEUE_PATH.unlink()
            except Exception:
                pass
    except Exception:
        pass

    return {'processed': len(items), 'succeeded': succeeded, 'failed': len(remaining), 'errors': errors}


def validate_webhook(url):
    # Try DNS resolve and a lightweight OPTIONS request to see allowed methods
    result = {'url': url, 'ok': False, 'details': ''}
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        # DNS
        try:
            socket.getaddrinfo(host, port)
        except Exception as e:
            result['details'] = f'DNS resolution failed: {e}'
            return result

        # OPTIONS
        try:
            req = urllib.request.Request(url, method='OPTIONS')
            with urllib.request.urlopen(req, timeout=5) as resp:
                allow = resp.getheader('Allow')
                result['ok'] = True
                result['details'] = f'OPTIONS OK; Allow: {allow}'
                return result
        except urllib.error.HTTPError as he:
            result['details'] = f'HTTPError {he.code}: {he.reason}'
            return result
        except urllib.error.URLError as ue:
            result['details'] = f'URLError: {ue.reason}'
            return result
    except Exception as e:
        result['details'] = f'Unexpected error: {e}'
        return result


def validate_smtp(smtp_cfg):
    result = {'ok': False, 'details': ''}
    if not smtp_cfg:
        result['details'] = 'No SMTP config'
        return result
    server = smtp_cfg.get('server')
    port = smtp_cfg.get('port', 25)
    try:
        # DNS
        try:
            socket.getaddrinfo(server, port)
        except Exception as e:
            result['details'] = f'DNS resolution failed: {e}'
            return result
        # TCP connect
        try:
            sock = socket.create_connection((server, port), timeout=5)
            sock.close()
            result['ok'] = True
            result['details'] = 'TCP connect successful'
            return result
        except Exception as e:
            result['details'] = f'TCP connect failed: {e}'
            return result
    except Exception as e:
        result['details'] = f'Unexpected error: {e}'
        return result
def main():
    st.title('Proposal Agent — Calibration Dashboard')

    st.sidebar.header('Data and Model')
    st.sidebar.write(f'Model: {MODEL_PATH}')
    st.sidebar.write(f'Historical: {HIST_PATH}')

    if not MODEL_PATH.exists():
        st.error('Model file missing: ' + str(MODEL_PATH))
        return
    if not HIST_PATH.exists():
        st.warning('Historical file missing: ' + str(HIST_PATH))

    model = json.loads(MODEL_PATH.read_text(encoding='utf-8'))
    df = load_rows(HIST_PATH) if HIST_PATH.exists() else pd.DataFrame()

    st.header('Historical statistics')
    if df.empty:
        st.info('No historical data available.')
    else:
        st.write('Historical sample count:', len(df))
        # show sample of data
        st.dataframe(df.head(50))

        overall = compute_median_ratio(df)
        st.write('Computed overall median(actual/estimated):', overall)

        products = df['product'].dropna().unique()
        prod_table = []
        for p in products:
            val = compute_median_ratio(df, p)
            prod_table.append({'product': p, 'calibration': val})
        st.table(pd.DataFrame(prod_table))

    st.header('Proposed updates')
    # editable per-product values default to computed or model
    proposed = {}
    for p in model.get('product_profiles', {}).keys():
        current = model.get('product_calibration', {}).get(p, model.get('calibration_factor', 1.0))
        computed = None
        if not df.empty:
            computed = compute_median_ratio(df, p)
        val = st.number_input(f'Calibration for {p}', value=float(current), format='%.4f')
        st.write('Computed from historical:', computed)
        proposed[p] = float(val)

    st.write('Customer adjustments (sample):')
    cust_adj = {}
    if CUSTOMER_ADJ.exists():
        cust_adj = json.loads(CUSTOMER_ADJ.read_text(encoding='utf-8'))
        st.json(cust_adj)
    else:
        st.info('No customer adjustments file found.')

    if st.button('Request apply proposed calibrations (create pending request)'):
        # write a pending request rather than apply directly
        try:
            req = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'requester': getpass.getuser(),
                'proposed': proposed,
                'model_snapshot': model
            }
            PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
            with PENDING_PATH.open('a', encoding='utf-8') as pf:
                pf.write(json.dumps(req) + '\n')
            st.success('Pending calibration request created — awaiting approval')
            # send notification (webhook or email) if configured
            try:
                send_notification('pending_created', req)
                st.info('Notification sent for pending request')
            except Exception as e:
                st.warning('Notification failed: ' + str(e))
                try:
                    enqueue_failed_notification('pending_created', req, [str(e)])
                    st.info('Failed notification queued for retry')
                except Exception:
                    pass
        except Exception as e:
            st.error('Failed to create pending request: ' + str(e))

    st.header('Pending approval requests')
    # determine if current user is an approver
    current_user = getpass.getuser()
    approvers = []
    if APPROVERS_PATH.exists():
        try:
            approvers = json.loads(APPROVERS_PATH.read_text(encoding='utf-8')).get('approvers', [])
        except Exception:
            approvers = []

    is_approver = current_user in approvers

    if PENDING_PATH.exists():
        pending_lines = [l for l in PENDING_PATH.read_text(encoding='utf-8').splitlines() if l.strip()]
        pending = [json.loads(l) for l in pending_lines]
        if not pending:
            st.info('No pending requests')
        else:
            for idx, req in enumerate(pending):
                st.subheader(f"Request {idx+1} by {req.get('requester')} at {req.get('timestamp')}")
                st.write('Proposed values:')
                st.json(req.get('proposed'))
                if is_approver:
                    if st.button(f"Approve request {idx+1}", key=f"approve_{idx}"):
                        # apply this pending request
                        try:
                            bak = MODEL_PATH.with_suffix('.json.approved.bak')
                            MODEL_PATH.rename(bak)
                            model['calibration_factor'] = float(median([(v) for v in req.get('proposed', {}).values()]))
                            model['product_calibration'] = req.get('proposed', {})
                            MODEL_PATH.write_text(json.dumps(model, indent=2), encoding='utf-8')
                            # write audit entry including approver
                            entry = {
                                'timestamp': datetime.utcnow().isoformat() + 'Z',
                                'requester': req.get('requester'),
                                'approver': getpass.getuser(),
                                'backup': str(bak.name),
                                'proposed': req.get('proposed'),
                                'previous_global': req.get('model_snapshot', {}).get('calibration_factor'),
                                'previous_product_calibration': req.get('model_snapshot', {}).get('product_calibration', {})
                            }
                            with AUDIT_PATH.open('a', encoding='utf-8') as af:
                                af.write(json.dumps(entry) + '\n')
                            # remove this pending line
                            remaining = [json.loads(l) for l in pending_lines]
                            remaining.pop(idx)
                            PENDING_PATH.write_text('\n'.join([json.dumps(r) for r in remaining]), encoding='utf-8')
                            st.success('Approved and applied request; audit entry written')
                            try:
                                send_notification('pending_approved', entry)
                                st.info('Notification sent for approval')
                            except Exception as e:
                                st.warning('Approval notification failed: ' + str(e))
                                try:
                                    enqueue_failed_notification('pending_approved', entry, [str(e)])
                                    st.info('Failed approval notification queued for retry')
                                except Exception:
                                    pass
                            st.experimental_rerun()
                        except Exception as e:
                            st.error('Failed to approve request: ' + str(e))
                else:
                    st.info('You are not an approver. Approval requires an approver account.')
    else:
        st.info('No pending requests file found')

    st.header('Manage Approvers')
    # Only existing approvers may update the approvers list
    if is_approver:
        st.write('Current approvers:')
        st.write(approvers)
        new_app = st.text_input('Add approver username', key='new_approver')
        if st.button('Add approver'):
            if not new_app:
                st.warning('Enter a username to add')
            elif new_app in approvers:
                st.info('User already an approver')
            else:
                approvers.append(new_app)
                try:
                    APPROVERS_PATH.write_text(json.dumps({'approvers': approvers}, indent=2), encoding='utf-8')
                    # audit
                    entry = {
                        'timestamp': datetime.utcnow().isoformat() + 'Z',
                        'actor': current_user,
                        'action': 'add_approver',
                        'target': new_app
                    }
                    with AUDIT_PATH.open('a', encoding='utf-8') as af:
                        af.write(json.dumps(entry) + '\n')
                    st.success(f'Added approver {new_app}')
                    st.experimental_rerun()
                except Exception as e:
                    st.error('Failed to add approver: ' + str(e))

        rem = None
        if approvers:
            rem = st.selectbox('Remove approver', options=approvers, key='remove_approver')
        if st.button('Remove approver'):
            if rem and rem in approvers:
                if rem == current_user:
                    st.warning('You cannot remove yourself')
                else:
                    approvers.remove(rem)
                    try:
                        APPROVERS_PATH.write_text(json.dumps({'approvers': approvers}, indent=2), encoding='utf-8')
                        entry = {
                            'timestamp': datetime.utcnow().isoformat() + 'Z',
                            'actor': current_user,
                            'action': 'remove_approver',
                            'target': rem
                        }
                        with AUDIT_PATH.open('a', encoding='utf-8') as af:
                            af.write(json.dumps(entry) + '\n')
                        st.success(f'Removed approver {rem}')
                        st.experimental_rerun()
                    except Exception as e:
                        st.error('Failed to remove approver: ' + str(e))
    else:
        st.info('Approver management restricted to approver accounts')

    st.header('Notifications (Admin)')
    cfg = load_notifications_config()
    st.write('Current notification configuration (env overrides shown):')
    st.json(cfg)

    if is_approver:
        st.subheader('Edit persisted notification settings (non-secret values only)')
        edit_enabled = st.checkbox('Persist notifications enabled', value=bool(cfg.get('enabled', False)))
        edit_webhooks = st.text_area('Persist webhook URLs (one per line)', value='\n'.join(cfg.get('webhooks', [])))
        if st.button('Save notification settings'):
            try:
                # write only non-secret fields to file; secrets should be provided via env vars
                out = {
                    'enabled': bool(edit_enabled),
                    'webhooks': [u.strip() for u in edit_webhooks.splitlines() if u.strip()]
                }
                NOTIFY_PATH.parent.mkdir(parents=True, exist_ok=True)
                NOTIFY_PATH.write_text(json.dumps(out, indent=2), encoding='utf-8')
                st.success('Persisted notification settings (secrets must remain in env)')
                st.experimental_rerun()
            except Exception as e:
                st.error('Failed to save settings: ' + str(e))
    else:
        st.info('Notification settings can be changed by approvers only')

    st.subheader('Test notifications')
    st.write('By default this is a dry-run. To actually send, set `PROPOSAL_NOTIFY_ENABLED=true` and provide SMTP/webhook secrets via environment variables.')
    test_payload = {'event': 'manual_test', 'payload': {'initiator': current_user, 'time': datetime.utcnow().isoformat() + 'Z'}}
    if st.button('Dry-run: show notification payload'):
        st.json(test_payload)
        st.info('Dry-run: no network calls made')

    send_confirm = st.text_input('Type YES to confirm sending live notification', value='', key='notif_confirm')
    if st.button('Send live test notification'):
        if send_confirm != 'YES':
            st.warning('Type YES to confirm live send')
        else:
            try:
                send_notification('manual_test', test_payload)
                st.success('Notification sent (check configured webhooks / SMTP recipients)')
            except Exception as e:
                st.error('Send failed: ' + str(e))
                try:
                    enqueue_failed_notification('manual_test', test_payload, [str(e)])
                    st.info('Failed test notification queued for retry')
                except Exception:
                    pass

    st.subheader('Queued notifications')
    qitems = read_notification_queue()
    st.write(f'Queued notifications: {len(qitems)}')
    if qitems:
        # show a brief preview of queued items
        preview = [{'timestamp': q.get('timestamp'), 'event': q.get('event'), 'attempts': q.get('attempts', 0)} for q in qitems]
        st.table(pd.DataFrame(preview))
        if st.button('Retry queued notifications'):
            st.info('Processing queued notifications...')
            res = process_notification_queue(send_notification)
            st.write('Processed:', res.get('processed'))
            st.write('Succeeded:', res.get('succeeded'))
            st.write('Remaining (failed):', res.get('failed'))
            if res.get('errors'):
                st.json(res.get('errors'))

    st.subheader('Diagnostics')
    st.write('Validate configured webhooks and SMTP connectivity')
    if st.button('Validate webhooks'):
        cfg = load_notifications_config()
        urls = cfg.get('webhooks', [])
        if not urls:
            st.info('No webhooks configured')
        else:
            rows = []
            for u in urls:
                r = validate_webhook(u)
                rows.append(r)
            st.json(rows)

    if st.button('Validate SMTP'):
        cfg = load_notifications_config()
        smtp_cfg = cfg.get('smtp')
        r = validate_smtp(smtp_cfg)
        st.json(r)

    st.header('Audit log')
    if AUDIT_PATH.exists():
        try:
            logs = [json.loads(l) for l in AUDIT_PATH.read_text(encoding='utf-8').splitlines() if l.strip()]
            # show recent 20 entries
            df_logs = pd.DataFrame(list(reversed(logs[-20:])))
            st.dataframe(df_logs)
        except Exception as e:
            st.warning('Failed to read audit log: ' + str(e))
    else:
        st.info('No audit entries yet.')


if __name__ == '__main__':
    main()
