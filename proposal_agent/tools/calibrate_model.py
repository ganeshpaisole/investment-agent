#!/usr/bin/env python3
"""Calibrate estimation_model.json from historical bids.

Expects CSV columns: product,estimated_total,actual_total
Computes calibration_factor = mean(actual_total / estimated_total)
and updates the model JSON (backing up original) when --update is provided.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def compute_calibration(rows, product_filter=None):
    ratios = []
    for r in rows:
        prod = r.get('product','').strip().lower()
        try:
            est = float(r.get('estimated_total', 0))
            act = float(r.get('actual_total', 0))
        except Exception:
            continue
        if est <= 0:
            continue
        if product_filter and prod != product_filter.lower():
            continue
        ratios.append(act / est)
    if not ratios:
        return None
    # use median to reduce outlier impact
    return median(ratios)


def load_rows(path: Path):
    with path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--historical', default='proposal_agent/data/historical_bids_example.csv')
    p.add_argument('--model', default='proposal_agent/data/estimation_model.json')
    p.add_argument('--product', help='Optional product key to compute product-level calibration')
    p.add_argument('--update', action='store_true', help='Write calibration factor into model JSON (backup created)')
    args = p.parse_args()

    hist = Path(args.historical)
    model_path = Path(args.model)
    if not hist.exists():
        print('Historical file not found:', hist)
        raise SystemExit(2)
    if not model_path.exists():
        print('Model file not found:', model_path)
        raise SystemExit(2)

    rows = load_rows(hist)
    overall = compute_calibration(rows, None)

    # compute per-product calibration (median)
    product_keys = set([r.get('product','').strip().lower() for r in rows if r.get('product')])
    product_cal = {}
    for pk in product_keys:
        val = compute_calibration(rows, pk)
        if val:
            product_cal[pk] = round(val, 4)

    # compute simple per-customer multipliers if 'customer' column present
    customer_adj = {}
    has_customer = any('customer' in r and r['customer'].strip() for r in rows)
    if has_customer:
        customers = set([r.get('customer','').strip() for r in rows if r.get('customer')])
        for c in customers:
            try:
                vals = [float(r.get('actual_total'))/float(r.get('estimated_total')) for r in rows if r.get('customer','').strip()==c and float(r.get('estimated_total',0))>0]
                if vals:
                    customer_adj[c] = round(median(vals),4)
            except Exception:
                continue

    print('Overall calibration factor (median actual/estimated):', overall)
    print('Per-product calibration factors:', product_cal)
    if customer_adj:
        print('Per-customer adjustments:', customer_adj)

    if args.update and overall:
        # backup
        bak = model_path.with_suffix('.json.bak')
        print('Backing up', model_path, '->', bak)
        bak.write_bytes(model_path.read_bytes())
        model = json.loads(model_path.read_text(encoding='utf-8'))
        model['calibration_factor'] = round(overall, 4)
        if product_cal:
            model['product_calibration'] = product_cal
        if customer_adj:
            # write customer adjustments into model under key 'customer_adjustments'
            model['customer_adjustments'] = customer_adj
        model_path.write_text(json.dumps(model, indent=2), encoding='utf-8')
        print('Updated model with calibration_factor and product/customer mappings')


if __name__ == '__main__':
    main()
