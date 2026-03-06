from nse_agent.utils.bse_parser import BSEFilingParser, COMPANY_BANK_PATTERNS


def test_company_bank_patterns_exported():
    assert isinstance(COMPANY_BANK_PATTERNS, dict)


def test_extract_banking_metrics_basic():
    p = BSEFilingParser("HDFCBANK")
    sample = (
        "Capital Adequacy Ratio 19.55% 19.97% 18.80% 19.55% 18.80%\n"
        "Net interest margin was at 3.54% on total assets\n"
        "CASA deposits were 38.50% of total deposits\n"
    )
    res = p._extract_banking_metrics(sample)
    assert isinstance(res, dict)
    # Expect core banking metrics to be present and numeric
    assert "car" in res and isinstance(res["car"], (int, float))
    assert "nim" in res and isinstance(res["nim"], (int, float))
    assert "casa" in res and isinstance(res["casa"], (int, float))
