"""Human and machine reports. Diagnostic text stays out of the JSON file's neighbor CSV."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from machineear.errors import MachineEarError


def write_outputs(result: dict[str, object], directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    root = directory.resolve()
    paths = {
        "result": _inside(root, "result.json"),
        "scores": _inside(root, "scores.csv"),
        "html": _inside(root, "summary.html"),
        "text": _inside(root, "summary.txt"),
    }
    paths["result"].write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_scores(paths["scores"], result)
    paths["html"].write_text(render_html(result), encoding="utf-8")
    paths["text"].write_text(render_text(result) + "\n", encoding="utf-8")
    return paths


def render_text(result: dict[str, object]) -> str:
    status = str(result["status"])
    lines = [
        "MachineEar calibration review",
        f"status: {status}",
        f"domain: {result['domain']}",
        f"anomaly_status_omitted: {str(result['anomaly_status_omitted']).lower()}",
    ]
    diagnostic = result.get("diagnostic")
    if isinstance(diagnostic, dict):
        lines.append(f"diagnostic: {diagnostic['code']}: {diagnostic['message']}")
    thresholds = result.get("thresholds") or {}
    if isinstance(thresholds, dict):
        lines.append("source threshold: " + _threshold_text(thresholds.get("source")))
        lines.append("proposed threshold: " + _threshold_text(thresholds.get("proposed")))
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        lines.extend(_metric_lines(metrics))
    splits = result.get("splits") or {}
    if isinstance(splits, dict) and splits:
        rendered = ", ".join(f"{name}={count}" for name, count in sorted(splits.items()))
        lines.append(f"split sizes: {rendered}")
    lines.append(f"recovery: {result.get('recovery', '')}")
    lines.append(f"manifest sha256: {result.get('manifest_sha256', '')}")
    lines.append(f"config sha256: {result.get('config_sha256', '')}")
    return "\n".join(lines)


def render_html(result: dict[str, object]) -> str:
    title = "MachineEar calibration review"
    domains = result.get("domains") or []
    selected = result.get("domain") if result.get("domain") not in {None, "all"} else (domains[0] if domains else "")
    options = "".join(
        f'<option value="{html.escape(str(name))}"{" selected" if name == selected else ""}>{html.escape(str(name))}</option>'
        for name in domains
    )
    if not options:
        options = '<option value="">no domain</option>'
    metrics_json = json.dumps(result.get("metrics_by_domain") or {})
    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    thresholds = result.get("thresholds") if isinstance(result.get("thresholds"), dict) else {}
    status_line = (
        f"Status: {html.escape(str(result['status']))}. "
        f"Anomaly status omitted: {html.escape(str(result['anomaly_status_omitted']))}."
    )
    source_line = html.escape(_threshold_text(thresholds.get("source")))
    proposed_line = html.escape(_threshold_text(thresholds.get("proposed")))
    signal_line = (
        f"Sample rate {html.escape(str(config.get('sample_rate_hz')))} Hz. "
        f"FFT length {html.escape(str(config.get('n_fft')))}. "
        f"Hop {html.escape(str(config.get('hop_length')))} samples."
    )
    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:focus {{ outline: 3px solid #0b3a82; outline-offset: 2px; }}
body {{ margin: 0; font-family: "Segoe UI", sans-serif; color: #17202a; background: #f4f7fb; }}
main {{ max-width: 960px; margin: 0 auto; padding: 16px; }}
p, li, td, th, pre {{ overflow-wrap: anywhere; }}
svg {{ max-width: 100%; height: auto; }}
pre {{ white-space: pre-wrap; }}
h1 {{ font-size: 1.4rem; }}
section {{ background: #fff; border: 1px solid #d5dde8; border-radius: 8px; padding: 12px 16px; margin: 12px 0; }}
label {{ font-weight: 600; }}
select {{ min-height: 44px; font-size: 1rem; margin-left: 8px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #e1e7ef; vertical-align: top; }}
.note {{ color: #243140; }}
.status {{ font-weight: 700; }}
@media (max-width: 480px) {{
  table, thead, tbody, th, td, tr {{ display: block; }}
  th {{ display: none; }}
}}
</style>
</head>
<body>
<main>
<h1>{title}</h1>
<p class="status">{status_line}</p>
<section>
<h2>Domain</h2>
<label for="domain">Microphone domain</label>
<select id="domain">{options}</select>
<p id="domain-status" role="status" aria-live="polite"></p>
<div id="rates"></div>
</section>
<section>
<h2>Thresholds</h2>
<p>Source: {source_line}</p>
<p>Proposed: {proposed_line}</p>
<p>Recall drop (source minus proposed, absolute): {html.escape(str(result.get("recall_drop")))}. Tolerance is in the config block.</p>
</section>
<section>
<h2>Sessions</h2>
{_session_table(result)}
</section>
<section>
<h2>Score distribution</h2>
{_distribution(result)}
</section>
<section>
<h2>Clips that changed classification</h2>
<p class="note">Timeline is clip order. These files are fixed clips, not a stream.</p>
{_timeline(result)}
</section>
<section>
<h2>Labeled normal alert rate</h2>
<pre id="official-metrics">{html.escape(_metrics_block(result.get("metrics")))}</pre>
</section>
<section>
<h2>Provenance, units, and limits</h2>
<ul>
<li>Schema {html.escape(str(result.get("schema_version")))}.</li>
<li>Manifest SHA-256 {html.escape(str(result.get("manifest_sha256")))}.</li>
<li>Config SHA-256 {html.escape(str(result.get("config_sha256")))}.</li>
<li>Score unit: {html.escape(str((result.get("config") or {}).get("score_units")))}.</li>
<li>{signal_line}</li>
{"".join(f"<li>{html.escape(str(item))}</li>" for item in result.get("limitations") or [])}
</ul>
<p class="note">Recovery: {html.escape(str(result.get("recovery")))}</p>
</section>
<section>
<h2>Clip table</h2>
{_clip_table(result)}
</section>
</main>
<script>
const metricsByDomain = {metrics_json};
const select = document.getElementById("domain");
const status = document.getElementById("domain-status");
function applyDomain() {{
  const domain = select.value;
  let shown = 0;
  document.querySelectorAll("[data-domain]").forEach((row) => {{
    const match = !domain || row.getAttribute("data-domain") === domain || row.getAttribute("data-role") === "source";
    row.hidden = !match;
    if (match && row.tagName === "TR") shown += 1;
  }});
  const metrics = metricsByDomain[domain];
  const rates = document.getElementById("rates");
  if (metrics) {{
    rates.textContent = "Domain " + domain + " target-test false-alert and recall figures follow in the status line and are stored in result.json.";
    status.textContent = "Showing domain " + domain + ". " + summarize(metrics);
  }} else {{
    rates.textContent = "";
    status.textContent = "Showing " + shown + " rows.";
  }}
}}
function summarize(metrics) {{
  const source = metrics.source_threshold || {{}};
  const proposed = metrics.proposed_threshold;
  const far = source.target_normal_false_alerts || {{}};
  if (!far.count) {{
    return "No labeled target-test clips in this domain.";
  }}
  let text = "Source threshold false alerts " + far.alerts + "/" + far.count + ".";
  if (proposed && proposed.target_normal_false_alerts) {{
    const after = proposed.target_normal_false_alerts;
    text += " Proposed false alerts " + after.alerts + "/" + after.count + ".";
  }}
  return text;
}}
select.addEventListener("change", applyDomain);
applyDomain();
</script>
</body>
</html>
"""
    return body


def _metrics_block(metrics: object) -> str:
    if not isinstance(metrics, dict):
        return "No rates were computed."
    return "\n".join(_metric_lines(metrics))


def _metric_lines(metrics: dict[str, object]) -> list[str]:
    lines = []
    sizes = metrics.get("split_sizes") or {}
    if isinstance(sizes, dict):
        lines.append(
            "target test size: "
            f"all={sizes.get('target_test')} normal={sizes.get('target_test_normal')} "
            f"anomaly={sizes.get('target_test_anomaly')} unknown={sizes.get('target_test_unknown')}"
        )
    lines.append("source " + _rate_text(metrics.get("source_threshold")))
    proposed = metrics.get("proposed_threshold")
    lines.append("proposed " + ("not proposed" if proposed is None else _rate_text(proposed)))
    lines.append(f"delta false-alert rate: {metrics.get('delta_false_alert_rate')}")
    lines.append(f"delta anomaly recall: {metrics.get('delta_anomaly_recall')}")
    return lines


def _rate_text(block: object) -> str:
    if not isinstance(block, dict):
        return "unavailable"
    far = block.get("target_normal_false_alerts") or {}
    recall = block.get("target_anomaly_recall") or {}
    return (
        f"false alerts {far.get('alerts')}/{far.get('count')} (rate {far.get('rate')}); "
        f"anomaly recall {recall.get('alerts')}/{recall.get('count')} (rate {recall.get('rate')})"
    )


def _threshold_text(payload: object) -> str:
    if not isinstance(payload, dict):
        return "not available"
    return (
        f"{payload.get('value')} version {payload.get('version')} "
        f"quantile {payload.get('quantile')} n={payload.get('n_calibration')} "
        f"accepted={payload.get('accepted')} reason={payload.get('reason')}"
    )


def _session_table(result: dict[str, object]) -> str:
    rows = result.get("sessions") or []
    if not isinstance(rows, list) or not rows:
        return "<p>No session summary.</p>"
    body = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        domain = html.escape(str(row.get("microphone_domain", "")))
        role = "source" if str(row.get("splits", [""])[0]).startswith("source") else "target"
        body.append(
            '<tr data-domain="{domain}" data-role="{role}"><td>{session}</td><td>{domain}</td>'
            "<td>{splits}</td><td>{n}</td><td>{normals}</td><td>{source}</td><td>{proposed}</td></tr>".format(
                domain=domain,
                role=role,
                session=html.escape(str(row.get("session_id"))),
                splits=html.escape(", ".join(row.get("splits") or [])),
                n=html.escape(str(row.get("n_clips"))),
                normals=html.escape(str(row.get("n_labeled_normal"))),
                source=html.escape(str(row.get("source_threshold_normal_alerts"))),
                proposed=html.escape(str(row.get("proposed_threshold_normal_alerts"))),
            )
        )
    return (
        "<table><thead><tr><th>Session</th><th>Domain</th><th>Splits</th><th>Clips</th>"
        "<th>Labeled normals</th><th>Source alerts</th><th>Proposed alerts</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _clip_table(result: dict[str, object]) -> str:
    clips = result.get("clips") or []
    if not isinstance(clips, list):
        return "<p>No clips.</p>"
    body = []
    for clip in clips:
        if not isinstance(clip, dict):
            continue
        diagnostic = clip.get("diagnostic")
        detail = ""
        if isinstance(diagnostic, dict):
            detail = f"{diagnostic.get('code')}: {diagnostic.get('message')}"
        domain = html.escape(str(clip.get("microphone_domain", "")))
        role = "source" if str(clip.get("split", "")).startswith("source") else "target"
        body.append(
            '<tr data-domain="{domain}" data-role="{role}"><td>{clip_id}</td><td>{domain}</td>'
            "<td>{split}</td><td>{label}</td><td>{score}</td><td>{source}</td><td>{proposed}</td>"
            "<td>{changed}</td><td>{detail}</td></tr>".format(
                domain=domain,
                role=role,
                clip_id=html.escape(str(clip.get("clip_id"))),
                split=html.escape(str(clip.get("split"))),
                label=html.escape(str(clip.get("label"))),
                score=html.escape(str(clip.get("score"))),
                source=html.escape(str(clip.get("source_alert"))),
                proposed=html.escape(str(clip.get("proposed_alert"))),
                changed=html.escape(str(clip.get("classification_changed"))),
                detail=html.escape(detail),
            )
        )
    return (
        "<table><thead><tr><th>Clip</th><th>Domain</th><th>Split</th><th>Label</th><th>Score</th>"
        "<th>Source alert</th><th>Proposed alert</th><th>Changed</th><th>Diagnostic</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _timeline(result: dict[str, object]) -> str:
    clips = result.get("clips") or []
    if not isinstance(clips, list):
        return "<p>No timeline.</p>"
    changed = [clip for clip in clips if isinstance(clip, dict) and clip.get("classification_changed") is True]
    if not changed:
        return "<p>No clip changed classification between the frozen threshold and the proposal.</p>"
    items = "".join(
        '<li data-domain="{domain}">{clip_id}: score {score}, source {source}, proposed {proposed}</li>'.format(
            domain=html.escape(str(clip.get("microphone_domain", ""))),
            clip_id=html.escape(str(clip.get("clip_id"))),
            score=html.escape(str(clip.get("score"))),
            source=html.escape(str(clip.get("source_alert"))),
            proposed=html.escape(str(clip.get("proposed_alert"))),
        )
        for clip in changed
    )
    return f"<ol>{items}</ol>"


def _distribution(result: dict[str, object]) -> str:
    clips = [
        clip
        for clip in result.get("clips") or []
        if isinstance(clip, dict) and isinstance(clip.get("score"), int | float)
    ]
    if not clips:
        return "<p>No scores to plot.</p>"
    scores = [float(clip["score"]) for clip in clips]
    low = min(scores)
    high = max(scores)
    span = high - low or 1.0
    bins = 12
    source = [0] * bins
    target = [0] * bins
    for clip in clips:
        index = min(bins - 1, int((float(clip["score"]) - low) / span * bins))
        if str(clip.get("split", "")).startswith("source"):
            source[index] += 1
        else:
            target[index] += 1
    peak = max(source + target + [1])
    width = 28
    parts = [
        '<svg role="img" aria-label="Histogram of source-domain and target-domain scores" width="100%" height="160" viewBox="0 0 420 160">',
        '<rect width="420" height="160" fill="#f8fafc"></rect>',
    ]
    for index, (src, tgt) in enumerate(zip(source, target, strict=True)):
        x = 20 + index * width
        src_h = 100 * src / peak
        tgt_h = 100 * tgt / peak
        parts.append(
            f'<rect x="{x}" y="{120 - src_h}" width="10" height="{src_h}" fill="#0b3a82"><title>Source bin {index}: {src}</title></rect>'
        )
        parts.append(
            f'<rect x="{x + 12}" y="{120 - tgt_h}" width="10" height="{tgt_h}" fill="#b45309"><title>Target bin {index}: {tgt}</title></rect>'
        )
    parts.append('<text x="20" y="145" font-size="12" fill="#0b3a82">Source domain</text>')
    parts.append('<text x="140" y="145" font-size="12" fill="#b45309">Target domain</text>')
    parts.append("</svg>")
    parts.append(
        f"<p>Score range {low:.3f} to {high:.3f}. Dark blue is the source domain. Brown is the target domain.</p>"
    )
    return "".join(parts)


def _write_scores(path: Path, result: dict[str, object]) -> None:
    clips = result.get("clips") or []
    fieldnames = [
        "clip_id",
        "session_id",
        "microphone_domain",
        "wav_path",
        "sha256",
        "split",
        "label",
        "approved_normal",
        "seed",
        "score",
        "source_alert",
        "proposed_alert",
        "classification_changed",
        "uncertain",
        "diagnostic_code",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if not isinstance(clips, list):
            return
        for clip in clips:
            if not isinstance(clip, dict):
                continue
            diagnostic = clip.get("diagnostic")
            code = diagnostic.get("code") if isinstance(diagnostic, dict) else ""
            writer.writerow(
                {
                    "clip_id": clip.get("clip_id"),
                    "session_id": clip.get("session_id"),
                    "microphone_domain": clip.get("microphone_domain"),
                    "wav_path": clip.get("wav_path"),
                    "sha256": clip.get("sha256"),
                    "split": clip.get("split"),
                    "label": clip.get("label"),
                    "approved_normal": clip.get("approved_normal"),
                    "seed": clip.get("seed"),
                    "score": "" if clip.get("score") is None else clip.get("score"),
                    "source_alert": _cell(clip.get("source_alert")),
                    "proposed_alert": _cell(clip.get("proposed_alert")),
                    "classification_changed": _cell(clip.get("classification_changed")),
                    "uncertain": clip.get("uncertain"),
                    "diagnostic_code": code,
                }
            )


def _cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).lower() if isinstance(value, bool) else str(value)


def _inside(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if root != path.parent:
        raise MachineEarError("path_escape", f"refusing to write {name} outside the output directory")
    return path
