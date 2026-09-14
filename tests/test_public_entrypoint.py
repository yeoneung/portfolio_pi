"""Check the experiment-only entry points without refitting any candidates."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
import run
import run_certificate


@pytest.mark.parametrize('benchmark,module_name', [
    ('direct', 'direct_residual_certificate'),
    ('welfare', 'portfolio_welfare'),
])
def test_certificate_adapter_redirects_only_report_path(
        benchmark, module_name, tmp_path, monkeypatch):
    calls = []
    core = SimpleNamespace(
        TABLE=Path('manuscript/tab_example.tex'),
        main=lambda argv: calls.append(argv),
    )
    monkeypatch.setitem(sys.modules, module_name, core)
    monkeypatch.setattr(run_certificate, 'ROOT', tmp_path)
    run_certificate.main([benchmark, '--verify-only'])
    assert calls == [['--verify-only']]
    assert core.TABLE == tmp_path / 'results/tables/tab_example.tex'
    assert core.TABLE.parent.is_dir()
    assert not (tmp_path / 'manuscript').exists()


def test_public_commands_need_no_paper_package():
    assert 'build-paper' not in run.COMMANDS
    assert 'package' not in run.COMMANDS
    for commands in run.COMMANDS.values():
        for command in commands:
            if command[0] != '-m':
                assert (ROOT / command[0]).is_file()
                assert not command[0].startswith(('manuscript/', 'packaging/'))
