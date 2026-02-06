import React, { useState } from "react";
import { uploadVideo, getStatus, getResults } from "./api";
import ClipResults from "./components/ClipResults";

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [videoLink, setVideoLink] = useState("");
  const [useLink, setUseLink] = useState(false);
  const [cropMode, setCropMode] = useState("none");
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState("");
  const [results, setResults] = useState([]);

  async function handleUpload() {
    if (!videoFile && !videoLink) return;

    setStatus("Processing, this might take a while...");

    const { job_id } = await uploadVideo(videoFile, videoLink, cropMode);
    setJobId(job_id);
    setStatus("processing");

    const interval = setInterval(async () => {
      const s = await getStatus(job_id);

      if (s.status === "completed") {
        clearInterval(interval);
        setStatus("completed");
        const r = await getResults(job_id);
        setResults(r);
      } else if (s.status === "failed") {
        clearInterval(interval);
        setStatus("failed ❌");
      }
    }, 4000);
  }

  return (
    <div style={{ padding: 40, fontFamily: "Arial, sans-serif" }}>
      <h1>🎥 Videotto Clip Selector</h1>

      {!jobId && (
        <div style={{ marginBottom: 30 }}>
          <div>
            <label>
              <input
                type="radio"
                checked={!useLink}
                onChange={() => setUseLink(false)}
              />
              Upload file
            </label>

            <label style={{ marginLeft: 20 }}>
              <input
                type="radio"
                checked={useLink}
                onChange={() => setUseLink(true)}
              />
              Paste link
            </label>
          </div>

          {!useLink && (
            <input
              type="file"
              accept="video/*"
              onChange={(e) => setVideoFile(e.target.files[0])}
              style={{ marginTop: 10 }}
            />
          )}

          {useLink && (
            <input
              type="text"
              placeholder="Paste Dropbox link here"
              value={videoLink}
              onChange={(e) => setVideoLink(e.target.value)}
              style={{ marginTop: 10, width: "60%" }}
            />
          )}

          <div style={{ marginTop: 15 }}>
            <label style={{ fontWeight: "bold" }}>
              Select a crop dimension for the output top 3 clips:
            </label>

            <select
              value={cropMode}
              onChange={(e) => setCropMode(e.target.value)}
              style={{ marginLeft: 10 }}
            >
              <option value="none">Original (No Crop)</option>
              <option value="vertical">Vertical (9:16)</option>
              <option value="square">Square (1:1)</option>
            </select>
          </div>

          <button
            onClick={handleUpload}
            style={{ marginLeft: 10, padding: "6px 12px", marginTop: 15 }}
          >
            Upload & Process
          </button>
        </div>
      )}

      {status && <p>Status: {status}</p>}

      {status === "completed" && <ClipResults clips={results} />}
    </div>
  );
}

export default App;
