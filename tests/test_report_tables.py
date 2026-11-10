"""Check financial units and upward rounding of numerical report exports."""
from decimal import Decimal
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'analysis'))
import report_tables as report


@pytest.fixture(scope='module', autouse=True)
def generated_tables(tmp_path_factory):
    original = report.TABLES
    report.TABLES = tmp_path_factory.mktemp('numeric-tables')
    try:
        report.build_tables()
        yield
    finally:
        report.TABLES = original


def test_upward_rounding_encloses_stored_binary_bound():
    for value in [.000000123456789, .0046856557, 13.344308234617182, 35.68777864686418, 966.1235385645142]:
        for places in (3, 6, 8):
            rounded = Decimal(report.upper(value, places))
            exact = Decimal.from_float(value)
            assert rounded >= exact
            assert rounded-exact < Decimal(1).scaleb(-places)


def test_highdim_table_matches_certified_bp_fields():
    data = json.loads((ROOT/'results/highdim_local/summary.json').read_text())
    lines = (report.TABLES/'tab_highdim.tex').read_text().splitlines()
    for row in data['main_cases']:
        line = next(s for s in lines if s.startswith(str(row['state_dimension'])+' &'))
        cells = [x.strip() for x in line.split('&')]
        for index, field in enumerate(('two_sided_pde_ce_upper_bp', 'hybrid_ce_upper_bp', 'time_only_hybrid_ce_upper_bp'), 1):
            assert cells[index] == report.upper(row[field])


def test_neural_numerical_and_continuous_status_are_distinct():
    text = (report.TABLES/'tab_neural_duality.tex').read_text()
    data = json.loads((ROOT/'results/neural_duality/summary.json').read_text())
    assert report.upper(data['rigorous_relative_ce_upper_bp']) in text
    assert 'Continuous certificate' in text and 'sampling CI' in text
    assert 'Numerical estimate' in text
    for name,slabs in (('coarse',75),('refined',139)):
        bound=json.loads((ROOT/f'results/neural_separate_evaluation/whole_space_{name}.json').read_bytes())
        row=next(line for line in text.splitlines() if f'({slabs} slabs)' in line)
        assert report.upper(bound['relative_ce_loss_upper_bp']) in row
        assert 'Continuous certificate' in row
