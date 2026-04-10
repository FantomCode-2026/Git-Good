from pathlib import Path
import json
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from digitaltwin.exporter import export_temple_simulation_state


def main():
    base_dir = BASE_DIR
    report_path = base_dir / "outputs" / "reports" / "video_report.json"

    if not report_path.exists():
        raise FileNotFoundError(f"Missing report: {report_path}")

    report = json.loads(report_path.read_text())
    summary = report.get("summary", {})
    alerts = report.get("alerts", [])

    metrics_csv = base_dir / "outputs" / "reports" / "video_report_metrics.csv"
    metrics_history = []
    if metrics_csv.exists():
        import csv

        with metrics_csv.open() as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parsed = {}
                for key, value in row.items():
                    try:
                        parsed[key] = float(value) if value not in {"", None} else value
                    except ValueError:
                        parsed[key] = value
                metrics_history.append(parsed)

    result = export_temple_simulation_state(
        output_dir=base_dir / "digitaltwin",
        summary=summary,
        metrics_history=metrics_history,
        alerts_history=alerts,
    )
    print(result["js_path"])


if __name__ == "__main__":
    main()
