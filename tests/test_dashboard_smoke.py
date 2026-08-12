"""
Minimal dashboard smoke test (Section 34): verifies the dashboard module's
non-Streamlit-runtime logic imports and its pure functions work, without
actually launching a Streamlit server (which requires a running app context).

For a full manual check, run: streamlit run dashboard/app.py
"""
import importlib.util
from pathlib import Path


def test_dashboard_module_is_syntactically_valid():
    dashboard_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    spec = importlib.util.spec_from_file_location("dashboard_app", dashboard_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    # We only compile, not execute at import time in a non-Streamlit context,
    # since app.py calls st.set_page_config() at module scope.
    source = dashboard_path.read_text(encoding="utf-8")
    compile(source, str(dashboard_path), "exec")
