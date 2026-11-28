# Raw data provenance

Audit of where the raw data of the 26 sessions comes from, whether the source files
still exist, and whether the copy on S3 is the same as the source.

- Audit dates: 2026-11-14 to 2026-11-21.
- Scope: `s3://hm2p-rawdata/rawdata/` against its two source folders on Dropbox.
- All audit operations on the source folders and on S3 were read-only.
- Update of 2026-11-21, after the audit: the one file found missing from S3 was
  uploaded (item 1 under "Differences and open items"). The tables and counts below
  describe the state before that upload.
- Update of 2026-11-22: the camera calibration files were added to the repository
  and to S3 (item 2).

## Summary

- All 26 sessions in `metadata/experiments.csv` have both source folders on Dropbox.
- 503 of the 504 files selected by the upload rule are on S3 with the same name and
  byte size. No S3 object is absent from Dropbox.
- 502 of the 503 S3 objects have the same content as the Dropbox file (checksum
  comparison). The one difference is a Finder `.DS_Store` file.
- One source file was absent from S3 at the time of the audit:
  `20221115_13_27_42_1118213_XYT.raw` (1.29 GB). It was uploaded on 2026-11-21 and
  its content was verified, so all 504 files of the upload set are now on S3.
- Versioning is enabled on the `hm2p-rawdata` bucket.

## Sources and mapping

Source root: `Dropbox/Neuro/Margrie/` (local path
`~/Library/CloudStorage/Dropbox/Neuro/Margrie/`). The mapping is defined in
`scripts/upload_to_s3.py`.

| Source folder (under `Margrie/`) | S3 prefix in `hm2p-rawdata` | Files skipped by the upload rule |
| --- | --- | --- |
| `shared/lab-108/experiments/01 lights-maze/{YYYY_MM_DD}/{exp_id}/` | `rawdata/sub-{animal_id}/ses-{YYYYMMDD}T{HHMMSS}/funcimg/` | `*_side_left.camera.mp4`, `*_XYT.red.tif` |
| `hm2p/video/{exp_id}/` | `rawdata/sub-{animal_id}/ses-{YYYYMMDD}T{HHMMSS}/behav/` | `*-undistort.mp4`, `*_side_left.camera.mp4` |

`exp_id` has the form `YYYYMMDD_HH_MM_SS_{animal_id}`. Subfolders are kept (for
example `behav/meta/`). The upload used `aws s3 sync` without `--delete`.

## Source folders (listing of 2026-11-21)

| | Files | Size |
| --- | --- | --- |
| All files in the 52 source folders | 548 | 131.06 GB |
| Selected by the upload rule | 504 | 99.48 GB |
| Skipped by the upload rule | 44 | 31.58 GB |

Skipped files exist only on Dropbox: 26 undistorted overhead videos (13.47 GB),
13 side-camera videos (10.35 GB) and 5 red-channel TIFFs (7.77 GB).

File types present per session, upload set only:

| File | Sessions (of 26) |
| --- | --- |
| `_XYT.tif` (2P movie) | 26 |
| `_XYT.ini`, `_OME.xml`, `.meta.txt` | 26 |
| `-di.tdms` (digital DAQ lines) | 26 |
| `-ai.tdms` (analogue DAQ lines) | 3 |
| `_XYT.raw` (SciScan raw copy) | 24 on Dropbox; absent for `20211216_14_36_39_1115816` and `20220531_11_06_13_1117217` |
| `overhead.camera.mp4` in the raw folder | 8 |
| `overhead.camera-cropped.mp4` | 26 |
| `meta.txt`, `meta/crop.csv`, `meta/scale.csv`, `meta/roi.csv`, `meta/movie-frame.tif` | 26 |

No file in the listing is empty and none is a symbolic link.

The video `meta.txt` of ten sessions was modified on 2026-01-03. The upload took
place afterwards, and the checksums confirm that S3 holds the modified versions.

## S3 bucket (listing of 2026-11-21)

- Bucket `hm2p-rawdata`, region `ap-southeast-2`, versioning `Enabled`.
- 645 objects, 107.11 GB.

| Prefix | Objects | Size | Uploaded (UTC) |
| --- | --- | --- | --- |
| `rawdata/*/*/funcimg/` | 325 | 94.91 GB | 2026-03-01 to 2026-03-02 |
| `rawdata/*/*/behav/` | 178 | 3.28 GB | 2026-03-01 to 2026-03-02 |
| `sourcedata/brains-sorted/` | 43 | 5.53 GB | 2026-03-09 |
| `sourcedata/zstacks/` (13 folders) | 97 | 3.40 GB | 2026-03-09 |
| `sourcedata/metadata/` | 2 | < 1 MB | 2026-03-02 |

Storage class is `STANDARD` for 416 objects and `STANDARD_IA` for 229 (mostly
the large TIFF, raw and video files). Downloads from `STANDARD_IA` carry a per-GB
retrieval charge. The bucket has one lifecycle rule, `ia-after-30d`, which moves
objects larger than 128 KB to `STANDARD_IA` 30 days after upload. The rule has no
expiration action.

## Comparison by name and size

| | Files | Size |
| --- | --- | --- |
| Upload set on Dropbox | 504 | 99.48 GB |
| Objects under `rawdata/` | 503 | 98.19 GB |
| Same key and same byte size | 503 | 98.19 GB |
| Same key, different size | 0 | |
| On S3, not on Dropbox | 0 | |
| On Dropbox (upload set), not on S3 | 1 | 1.29 GB |

## Comparison by content

Method: the MD5 of each Dropbox file was computed locally and compared with the
ETag that S3 recorded at upload. For single-part uploads the ETag is the MD5 of the
file. For multipart uploads (112 of the 503 objects) the ETag is the MD5 of the
concatenated per-part MD5 digests followed by `-{number of parts}`; it was recomputed
with 8 MiB parts, the AWS CLI default, which reproduced every multipart ETag. No
object was downloaded from S3.

| Run | Date | Files | Size | Match | Not matching |
| --- | --- | --- | --- | --- | --- |
| Files under 5 MB, and all TDMS files | 2026-11-16 | 420 | 2.40 GB | 419 | 1 (`.DS_Store`) |
| Large files stored locally | 2026-11-16 | 29 | 44.59 GB | 29 | 0 |
| Large files that were online-only | 2026-11-16 to 2026-11-17 | 54 | 51.20 GB | 48 | 6 not read |
| Retry of the 6 unread files | 2026-11-21 | 6 | 7.74 GB | 6 | 0 |
| **Total** | | **503** | **98.19 GB** | **502** | **1** |

- The six files not read in the third run were `_XYT.raw` files whose Dropbox
  download was cancelled (`ECANCELED`). All six matched when read again.
- The mismatching file is `behav/.DS_Store` of session `20221115_13_27_42_1118213`,
  modified on Dropbox on 2026-03-09, after the upload. It is a Finder metadata file.
  24 `.DS_Store` files are on S3 because the upload rule does not skip them.
- Every `_XYT.tif`, cropped video, raw overhead video, TDMS file and metadata file
  of all 26 sessions matched.

Per-file checksum rows were retained only for the last two runs (54 files,
`data-provenance/checksum_results_per_file_2026-11-21.tsv`). The per-file output of
the first two runs was held in a temporary folder that was cleared before it was
saved; for those 449 files this document records the totals only. The S3 listing
below contains the ETag of every object, so the comparison can be repeated for any
file.

## Differences and open items

1. **File not on S3.** `20221115_13_27_42_1118213_XYT.raw` (1,290,240,000 bytes) is in
   the upload set and on Dropbox but not on S3. The `_XYT.tif` of the same session is
   on S3 and verified. The cause was not determined.

   **Resolved 2026-11-21.** The file was uploaded with `aws s3 cp` to
   `rawdata/sub-1118213/ses-20221115T132742/funcimg/20221115_13_27_42_1118213_XYT.raw`.
   Before the upload the key held no object, no earlier version and no incomplete
   multipart upload. After the upload the object has 1,290,240,000 bytes and the
   ETag `fef6cf254fc6ce35f104a2e2447b2927-154`, equal to the 8 MiB multipart ETag
   computed from the Dropbox file. `rawdata/` now holds 504 objects. The listing
   table of 2026-11-21 predates the upload and does not contain this key.
2. **Camera calibration files.** `docs/data-guide.md` and `CLAUDE.md` state that the
   lens calibration `.npz` files were copied to `sourcedata/calibration/` in the
   repository and on S3. Neither location exists. The two files
   (`acA1300-200um_C125-0418-5M.npz`, `acA1300-200um_C125-0618-5M.npz`) are in the
   legacy repository under `cam-calibrations/` and in `old-pipeline/cam-calibrations/`.
   The current pipeline reads the cropped videos and does not load them.

   **Resolved 2026-11-22.** Both files were copied from the legacy repository to
   `sourcedata/calibration/`, where they are tracked in git, and uploaded to
   `s3://hm2p-rawdata/sourcedata/calibration/`. The keys held no object and no
   earlier version before the upload, and the upload was conditional on the key
   being absent. The S3 ETags equal the MD5 of the local files.

   | File | Bytes | SHA-256 |
   | --- | --- | --- |
   | `acA1300-200um_C125-0418-5M.npz` | 2,700 | `dcaa6acd7393ebbd220f9e03ef581719943951779a1fa2951736853331b566d3` |
   | `acA1300-200um_C125-0618-5M.npz` | 3,180 | `4076a6890d1d060de56da006cc5d9be6f74089c6996ce13e9ae11ad1f6657005` |

   The SHA-256 values are the same for the copies in the legacy repository, in
   `old-pipeline/cam-calibrations/` and in `sourcedata/calibration/`.
3. **`experiments.csv` on S3 is out of date.** `sourcedata/metadata/animals.csv` on S3
   is identical to `metadata/animals.csv` in the repository.
   `sourcedata/metadata/experiments.csv` on S3 (2,998 bytes, uploaded 2026-03-02)
   differs from the repository file (3,264 bytes), which has been edited since. The
   repository file is the reference.
4. **Data volume.** The figure of about 113 GB in `CLAUDE.md` was an estimate. The
   bucket holds 107.11 GB, of which 98.19 GB is session data.
5. **Not covered by this audit.** `sourcedata/brains-sorted/` and `sourcedata/zstacks/`
   were not compared with their Dropbox sources. The `hm2p-derivatives` bucket was
   not examined. The 44 files skipped by the upload rule have no copy on S3.

## Tables

In `docs/data-provenance/`:

| File | Content |
| --- | --- |
| `dropbox_source_listing_2026-11-21.tsv` | One row per file in the 52 source folders: path relative to `Margrie/`, size, modification time, expected S3 key, whether the upload rule selects it |
| `s3_rawdata_listing_2026-11-21.tsv` | One row per object in `hm2p-rawdata`: key, size, upload time, ETag, storage class |
| `checksum_results_per_file_2026-11-21.tsv` | Per-file checksum result for the 54 files of the last two runs |

## Repeating the checks

1. List each session's two source folders by name and size (`os.scandir` and
   `os.lstat`; no file is opened, so online-only Dropbox files are not downloaded).
2. `aws s3api get-bucket-versioning --bucket hm2p-rawdata`
3. `aws s3api list-objects-v2 --bucket hm2p-rawdata` for keys, sizes and ETags.
4. Compare keys and sizes using the mapping above.
5. For content, compute the MD5 (or the 8 MiB multipart ETag) of each source file
   and compare it with the listed ETag. Reading an online-only Dropbox file
   downloads it; the full upload set is 99.48 GB.

`scripts/verify_s3_upload.sh` covers a subset of this: session presence, object
counts, the MD5 of `behav/meta.txt` and the size of the TDMS files.
