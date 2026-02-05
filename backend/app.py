
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os, shutil, uuid, requests, subprocess
from processor import process_video

app = FastAPI(title="Videotto Clip Selector")

# Folders
os.makedirs("exports", exist_ok=True)
os.makedirs("videos", exist_ok=True)

# Serve exported clips
app.mount("/exports", StaticFiles(directory="exports"), name="exports")

# In-memory job tracker
jobs = {}

# CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
def upload_video(
    file: UploadFile = File(None),
    link: str = Query(None),
    crop_mode: str = Query("none", enum=["none", "vertical", "square"])
):
    job_id = str(uuid.uuid4())
    file_path = f"videos/{job_id}.mp4"

    # ---- OPTION 1: FILE UPLOAD ----
    if file:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # ---- OPTION 2: LINK ----
    elif link:
        print("Downloading from link:", link)

        if "dropbox.com" in link:
            # Convert to direct download
            if "dl=0" in link:
                link = link.replace("dl=0", "dl=1")
            elif "dl=1" not in link:
                link = link + "?dl=1"

            print("Final Dropbox download URL:", link)

            r = requests.get(link, stream=True)
            if r.status_code != 200:
                print("Dropbox error:", r.status_code)
                return {"error": "Failed to download from Dropbox"}

            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        else:
            return {"error": "Unsupported link format"}

    else:
        return {"error": "No file or link provided"}

    # ---- PROCESSING (SYNC for simplicity) ----
    jobs[job_id] = {"status": "processing"}

    try:
        results = process_video(file_path, crop_mode, job_id)
        jobs[job_id] = {"status": "completed", "results": results}
    except Exception as e:
        print("ERROR DURING PROCESSING:", e)
        jobs[job_id] = {"status": "failed", "error": str(e)}

    return {"job_id": job_id}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"status": "not found"}
    return {"status": job["status"]}

@app.get("/results/{job_id}")
def get_results(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}
    if job["status"] != "completed":
        return {"error": f"Job is {job['status']}"}
    return job["results"]
