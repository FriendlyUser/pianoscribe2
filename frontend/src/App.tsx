import { useCallback, useEffect, useState } from "react";
import { ApiError, getConfig, watchJob } from "./api";
import { JobProgress } from "./components/JobProgress";
import { ScoreViewer } from "./components/ScoreViewer";
import { SubmitForm } from "./components/SubmitForm";
import { forgetJob, loadRecent, rememberJob, type RecentJob } from "./recent";
import { DEFAULT_CONFIG, type ClientConfig, type Job, type Submission } from "./types";

const jobIdFromHash = () => /^#\/job\/([0-9a-f]{32})$/.exec(window.location.hash)?.[1] ?? null;

export function App() {
  const [config, setConfig] = useState<ClientConfig>(DEFAULT_CONFIG);
  const [jobId, setJobId] = useState<string | null>(jobIdFromHash);
  const [job, setJob] = useState<Job | null>(null);
  const [cached, setCached] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [recent, setRecent] = useState<RecentJob[]>(loadRecent);

  useEffect(() => {
    getConfig().then(setConfig).catch(() => undefined);
    const onHash = () => setJobId(jobIdFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    setLoadError(null);
    if (!jobId) {
      setJob(null);
      return;
    }
    return watchJob(
      jobId,
      (next) => {
        setJob(next);
        if (next.status === "succeeded") setRecent(rememberJob(next.id, next.title));
      },
      (error: ApiError) => {
        setLoadError(error.status === 404 ? "This transcription has expired or does not exist." : error.message);
        if (error.status === 404) setRecent(forgetJob(jobId));
      },
    );
  }, [jobId]);

  const onSubmitted = useCallback((submission: Submission) => {
    setJob(submission.job);
    setCached(submission.cached);
    window.location.hash = `#/job/${submission.job.id}`;
  }, []);

  const reset = useCallback(() => {
    setCached(false);
    window.location.hash = "";
  }, []);

  const active = jobId !== null;
  return (
    <div className="shell">
      <header className="masthead">
        <a className="brand" href="#" onClick={reset} aria-label="PianoScribe home">
          <span className="brand-mark" aria-hidden="true">
            ♪
          </span>
          PianoScribe
        </a>
        {active && (
          <button type="button" className="button ghost" onClick={reset}>
            New transcription
          </button>
        )}
      </header>

      <main>
        {!active && (
          <>
            <section className="intro">
              <h1>Turn a recording into piano sheet music</h1>
              <p>
                Upload audio or video, or paste a YouTube link. Solo piano is written down note for note; a song with a
                band or vocals is arranged for piano. Either way you get a score to play along with, print or download.
              </p>
            </section>
            <SubmitForm config={config} onSubmitted={onSubmitted} />
            {recent.length > 0 && (
              <section className="recent" aria-labelledby="recent-heading">
                <h2 id="recent-heading">Recent scores</h2>
                <ul>
                  {recent.map((item) => (
                    <li key={item.id}>
                      <a href={`#/job/${item.id}`}>{item.title}</a>
                      <time dateTime={new Date(item.at).toISOString()}>{new Date(item.at).toLocaleDateString()}</time>
                    </li>
                  ))}
                </ul>
              </section>
            )}
          </>
        )}

        {active && loadError && (
          <section className="card notice error" role="alert">
            <h2>Transcription unavailable</h2>
            <p>{loadError}</p>
            <button type="button" className="button primary" onClick={reset}>
              Start a new transcription
            </button>
          </section>
        )}
        {active && !loadError && !job && <section className="card loading">Loading…</section>}
        {active && !loadError && job && job.status !== "succeeded" && <JobProgress job={job} onReset={reset} />}
        {active && !loadError && job && job.status === "succeeded" && <ScoreViewer job={job} cached={cached} />}
      </main>

      <footer className="footnote">
        Files are deleted after processing; scores are kept for{" "}
        {Math.round(config.result_retention_seconds / 86400)} days. Only transcribe music you have the right to use.
      </footer>
    </div>
  );
}
