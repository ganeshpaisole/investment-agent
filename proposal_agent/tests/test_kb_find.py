from proposal_agent.kb import loader


def test_find_clauses_by_keyword_matches_name_and_content(tmp_path):
    # create fake clauses dict
    clauses = {
        'finance_clause': 'This clause mentions finance and ledger operations.',
        'hr_clause': 'Human resources and payroll details.'
    }
    matches = loader.find_clauses_by_keyword(clauses, 'finance')
    assert 'finance_clause' in matches
    matches2 = loader.find_clauses_by_keyword(clauses, 'payroll')
    assert 'hr_clause' in matches2
