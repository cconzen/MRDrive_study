"""trim_vids_and_logs.py: trims the study videos and corresponding log files to when the start button is actually pressed (noted down in start_times.csv)."""

from pathlib import Path
import shutil
import subprocess
import pandas as pd

# =========================
# CONFIG
# =========================
ROOT = Path("../DATA/videologs")
START_TIMES_CSV = Path("../DATA/start_times.csv")
OUTPUT_ROOT = Path("../DATA/trimmed_data")

VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv"]
VIDEO_REENCODE = True   # True = more accurate, slower; False = faster, less exact
COPY_NONMATCHING_FILES = False  # copy other files from participant folders too

# crashed? skip these
SKIP_PARTICIPANTS = {
    "1", "10", "11", "12", "13",
    "15", "16", "17", "20", "21",
    "23", "24"
}

# =========================
# HELPERS
# =========================
def get_ordered_csvs(participant_dir: Path):
    csvs = list(participant_dir.glob("*.csv"))

    # exclude start_times or already-generated summaries if they live in ROOT
    csvs = [p for p in csvs if p.name != "start_times.csv"]

    if len(csvs) != 4:
        print(f"Warning: {participant_dir.name} has {len(csvs)} csvs, expected 4. Skipping participant.")
        return None

    lengths = []
    for csv_path in csvs:
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                n = sum(1 for _ in f)
        except UnicodeDecodeError:
            with open(csv_path, "r", encoding="latin-1") as f:
                n = sum(1 for _ in f)
        lengths.append((csv_path, n))

    lengths.sort(key=lambda x: x[1])
    return [x[0] for x in lengths]


def find_matching_video(csv_path: Path) -> Path | None:
    stem = csv_path.stem

    if not stem.startswith("varjo_gaze_output_"):
        print(f"Warning: unexpected csv name format: {csv_path.name}")
        return None

    suffix = stem.replace("varjo_gaze_output_", "", 1)

    for ext in VIDEO_EXTENSIONS:
        candidate = csv_path.parent / f"varjo_capture_{suffix}{ext}"
        if candidate.exists():
            return candidate
        candidate_upper = csv_path.parent / f"varjo_capture_{suffix}{ext.upper()}"
        if candidate_upper.exists():
            return candidate_upper

    wildcard_matches = list(csv_path.parent.glob(f"varjo_capture_{suffix}.*"))
    if wildcard_matches:
        return wildcard_matches[0]

    print(f"Warning: no matching video found for {csv_path.name}")
    return None


def trim_csv_by_time(csv_path: Path, output_path: Path, start_time_s: float):
    df = pd.read_csv(csv_path)

    if "relative_to_unix_epoch_timestamp" not in df.columns:
        raise ValueError(f"{csv_path} is missing 'relative_to_unix_epoch_timestamp'")

    t_ns = df["relative_to_unix_epoch_timestamp"].astype("float64")
    time_s = (t_ns - t_ns.iloc[0]) / 1e9

    trimmed = df[time_s > start_time_s].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trimmed.to_csv(output_path, index=False)

    return len(df), len(trimmed)


def trim_video_ffmpeg(input_path: Path, output_path: Path, start_time: float, reencode: bool = True):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = max(0.0, float(start_time))

    if reencode:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time),
            "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "aac",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time),
            "-i", str(input_path),
            "-c", "copy",
            str(output_path),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed for {input_path.name}\n{result.stderr}")


def maybe_copy_other_files(src_dir: Path, dst_dir: Path, files_to_skip: set[Path]):
    if not COPY_NONMATCHING_FILES:
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    for p in src_dir.iterdir():
        if p in files_to_skip:
            continue
        if p.is_file():
            shutil.copy2(p, dst_dir / p.name)



# =========================
# MAIN
# =========================
def main():
    if not START_TIMES_CSV.exists():
        raise FileNotFoundError(f"Could not find {START_TIMES_CSV}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    start_times_df = pd.read_csv(START_TIMES_CSV)
    start_times_df["participant_id"] = start_times_df["participant_id"].astype(str)

    participant_dirs = [p for p in ROOT.iterdir() if p.is_dir()]

    for participant_dir in sorted(participant_dirs):
        participant_id = participant_dir.name

        if participant_id in SKIP_PARTICIPANTS:
            print(f"\nSkipping participant {participant_id} (already processed)")
            continue
        print("\n" + "=" * 80)
        print(f"Participant {participant_id}")
        print("=" * 80)

        start_row = start_times_df[start_times_df["participant_id"] == participant_id]
        if start_row.empty:
            print(f"No start times found for participant {participant_id}, skipping.")
            continue
        start_row = start_row.iloc[0]

        ordered_csvs = get_ordered_csvs(participant_dir)
        if ordered_csvs is None:
            continue

        block_mapping = [
            ("trial", ordered_csvs[0], float(start_row["trial"])),
            ("Block1", ordered_csvs[1], float(start_row["Block1"])),
            ("Block2", ordered_csvs[2], float(start_row["Block2"])),
            ("Block3", ordered_csvs[3], float(start_row["Block3"])),
        ]

        participant_out_dir = OUTPUT_ROOT / participant_id
        participant_out_dir.mkdir(parents=True, exist_ok=True)

        used_files = set()

        for block_name, csv_path, start_time in block_mapping:
            print(f"\n{block_name}")
            print(f"  CSV:   {csv_path.name}")
            print(f"  Start: {start_time:.3f}s")

            used_files.add(csv_path)

            trimmed_csv_name = csv_path.stem + "_trimmed.csv"
            trimmed_csv_path = participant_out_dir / trimmed_csv_name

            try:
                n_before, n_after = trim_csv_by_time(csv_path, trimmed_csv_path, start_time)
                print(f"  Trimmed CSV rows: {n_before} -> {n_after}")
            except Exception as e:
                print(f"  Error trimming CSV {csv_path.name}: {e}")
                continue

            video_path = find_matching_video(csv_path)

            if video_path is None:
                continue

            used_files.add(video_path)

            trimmed_video_name = video_path.stem + "_trimmed" + video_path.suffix
            trimmed_video_path = participant_out_dir / trimmed_video_name

            print(f"  Video: {video_path.name}")

            try:
                trim_video_ffmpeg(
                    input_path=video_path,
                    output_path=trimmed_video_path,
                    start_time=start_time,
                    reencode=VIDEO_REENCODE
                )
                print(f"  Saved trimmed video: {trimmed_video_path}")
            except Exception as e:
                print(f"  Error trimming video {video_path.name}: {e}")

        maybe_copy_other_files(participant_dir, participant_out_dir, used_files)

    print("\nDone.")
    print(f"Trimmed files saved under: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()