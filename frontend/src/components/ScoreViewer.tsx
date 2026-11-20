import { useState } from "react";
import { artifactUrl } from "../api";
import { shell } from "../desktop";
import type { Job } from "../types";
import { formatBytes } from "../validation";

const DOWNLOADS: { name: string; label: string; primary?: boolean }[] = [
  { name: "score.pdf", label: "PDF", primary: true },
  { name: "score.html", label: "HTML" },
  { name: "score.musicxml", label: "MusicXML" },
  { name: "score.mid", label: "MIDI (score)" },
  { name: "performance.mid", label: "MIDI (performance)" },
  { name: "transcription.mid", label: "MIDI (all instruments)" },
];


function formatDuration(seconds: number): string {
  const whole = Math.round(seconds);
  return whole < 90 ? `${whole} s` : `${Math.floor(whole / 60)} min ${whole % 60} s`;
}

interface Props {
  job: Job;
  cached: boolean;
  /** Desktop app: the PDF is printed by the shell rather than downloaded from the server. */
  shellPdf?: boolean;
}

export function ScoreViewer({ job, cached, shellPdf = false }: Props) {
  const [pdfNote, setPdfNote] = useState<string | null>(null);
  const bridge = shellPdf ? shell() : undefined;
  const savePdf = async () => {
    if (!bridge) return;
    setPdfNote("Preparing PDF…");
    const outcome = await bridge.savePdf(job.id, job.title);
    setPdfNote(outcome.error ? `PDF failed: ${outcome.error}` : outcome.saved ? `Saved to ${outcome.path}` : null);
  };
  const result = job.result;
  if (!result) return null;
  const total = Object.values(job.timings).reduce((sum, value) => sum + value, 0);
  return (
    <section className="viewer">
      <div className="card viewer-head">
        <div>
          <h2>{job.title}</h2>
          <ul className="facts">
            <li>{result.key}</li>
            <li>{result.time_signature}</li>
            <li title={`${result.tempo_unit} notes per minute`}>
              {Math.round(result.tempo_bpm)} bpm
            </li>
            <li>{result.measures} bars</li>
            <li>
              {result.pages} {result.pages === 1 ? "page" : "pages"}
            </li>
            {result.mode === "arrangement" ? (
              <li title="The whole song (melody, bass, harmony) rewritten as a part one pianist can play.">
                Arranged for piano
              </li>
            ) : (
              <li title="The notes the pianist played, written down.">Piano transcription</li>
            )}
            {result.melody_fidelity !== null && (
              <li title="Share of the melody notes heard in the recording that the arrangement plays at the same moment.">
                {Math.round(result.melody_fidelity * 100)}% of the melody kept
              </li>
            )}
          </ul>
          <p className="hint">
            {cached ? "Served instantly from cache." : `Ready in ${formatDuration(total)}.`} Click any note to play
            from there.
          </p>
          {pdfNote && <p className="hint">{pdfNote}</p>}
        </div>
        <div className="downloads" aria-label="Downloads">
          {bridge && (
            <button type="button" className="button primary" onClick={savePdf}>
              PDF
            </button>
          )}
          {DOWNLOADS.filter((d) => d.name in result.artifacts).map((d) => (
            <a
              key={d.name}
              className={`button${d.primary ? " primary" : ""}`}
              href={artifactUrl(job.id, d.name, true)}
              download
              title={formatBytes(result.artifacts[d.name] ?? 0)}
            >
              {d.label}
            </a>
          ))}
        </div>
      </div>
      <iframe
        className="score-frame"
        title={`Sheet music: ${job.title}`}
        src={`${artifactUrl(job.id, "score.html")}?embed=1`}
        sandbox="allow-scripts allow-modals"
      />
    </section>
  );
}
