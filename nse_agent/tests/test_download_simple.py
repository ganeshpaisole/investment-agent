from nse_agent.download_and_parse_attach import simple_extract_metrics


def test_simple_extract_metrics_returns_values():
    sample = (
        "Capital Adequacy Ratio 18.80%\n"
        "NIM: 3.20%\n"
        "CASA deposits were 35.60% of total deposits\n"
    )
    res = simple_extract_metrics(sample)
    assert isinstance(res, dict)
    assert res.get('car') is not None
    assert res.get('nim') is not None
    assert res.get('casa') is not None
