
const API_BASE = "http://localhost:8000";

export async function uploadVideo(file, link, cropMode = "none") {
  const formData = new FormData();

  if (file) {
    formData.append("file", file);
  }

  const url = link
    ? `${API_BASE}/upload?crop_mode=${cropMode}&link=${encodeURIComponent(link)}`
    : `${API_BASE}/upload?crop_mode=${cropMode}`;

  const res = await fetch(url, {
    method: "POST",
    body: formData,
  });

  return res.json();
}


export async function getStatus(jobId) {
  const res = await fetch(`${API_BASE}/status/${jobId}`);
  return res.json();
}

export async function getResults(jobId) {
  const res = await fetch(`${API_BASE}/results/${jobId}`);
  return res.json();
}

