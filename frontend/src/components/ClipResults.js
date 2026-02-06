function ClipResults({ clips }) {
  return (
    <div>
      <h2>Top Clips</h2>
      {clips.map((c) => (
        <div key={c.clip} style={{ marginBottom: 20 }}>
          <h3>Clip {c.clip}</h3>
          <p>
            {c.start} → {c.end} ({c.duration})
          </p>
          <video
            src={`http://localhost:8000${c.file}`}
            controls
            width="400"
          />
          <p>{c.reason}</p>
        </div>
      ))}
    </div>
  );
}

export default ClipResults;
