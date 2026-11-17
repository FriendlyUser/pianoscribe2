import { useEffect, useState } from "react";
import type { Job, Stage } from "../types";

const SONG_STEPS: { stage: Stage; label: string }[] = [
  { stage: "ingesting", label: "Downloading and validating" },
  { stage: "separating", label: "Separating the instruments" },
  { stage: "transcribing", label: "Transcribing every part" },
  { stage: "arranging", label: "Arranging for piano" },
  { stage: "engraving", label: "Writing the score" },
  { stage: "ready", label: "Ready" },
];
const SOLO_STEPS: { stage: Stage; label: string }[] = [
  { stage: "ingesting", label: "Downloading and validating" },
  { stage: "transcribing", label: "Transcribing the piano" },
  { stage: "arranging", label: "Finding beats and bar lines" },
  { stage: "engraving", label: "Writing the score" },
  { stage: "ready", label: "Ready" },
];
const ORDER: Stage[] = ["queued", "ingesting", "separating", "transcribing", "arranging", "engraving", "ready"];

function useElapsed(since: number, running: boolean): string {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [running]);
  const seconds = Math.max(0, Math.floor(now / 1000 - since));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

interface Props {
  job: Job;
  onReset: () => void;
}

export function JobProgress({ job, onReset }: Props) {
  const failed = job.status === "failed";
  const elapsed = useElapsed(job.created_at, !failed);
  const reached = ORDER.indexOf(job.stage);

  if (failed) {
    return (
      <section className="card notice error" role="alert">
        <h2>We could not transcribe this</h2>
        <p>{job.error?.message ?? "Something went wrong while processing this audio."}</p>
        <p className="hint">
          {job.title} · reference {job.id.slice(0, 8)}
          {job.error?.code ? ` · ${job.error.code}` : ""}
        </p>
        <button type="button" className="button primary" onClick={onReset}>
          Try another file or link
        </button>
      </section>
    );
  }

  return (
    <section className="card progress" aria-live="polite">
      <div className="progress-head">
        <div>
          <h2>{job.title}</h2>
          <p className="hint">{job.status === "queued" ? "Waiting for a free worker…" : job.message}</p>
        </div>
        <span className="elapsed" aria-label="Elapsed time">
          {elapsed}
        </span>
      </div>
      <progress className="overall" max={1} value={job.progress} aria-label="Overall progress" />
      <ol className="steps">
        {(job.options.solo_piano ? SOLO_STEPS : SONG_STEPS).map((step) => {
          const index = ORDER.indexOf(step.stage);
          const state = reached > index ? "done" : reached === index ? "current" : "pending";
          const label = step.label;
          return (
            <li key={step.stage} className={state} aria-current={state === "current" ? "step" : undefined}>
              <span className="dot" aria-hidden="true" />
              {label}
            </li>
          );
        })}
      </ol>
      <p className="hint">
        You can leave this page open or come back later with the same link. A four-minute song takes about ten minutes on
        a mid-range GPU; without a GPU it takes far longer.
      </p>
    </section>
  );
}
