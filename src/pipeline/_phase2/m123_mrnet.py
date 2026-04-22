"""M-123: MRNet knee MRI abnormality/ACL/meniscus (3 binary labels, 3 views)."""
from __future__ import annotations
from pathlib import Path
import numpy as np, cv2, pandas as pd, zipfile
from common import DATA_ROOT, write_task, COLORS, fit_square, window_minmax, to_rgb

PID="M-123"; TASK_NAME="mrnet_knee_mri_abnormality"; FPS=5
PROMPT=("This is a knee MRI volume from the MRNet dataset (axial/coronal/sagittal views). "
        "Identify whether this knee shows any abnormality, anterior cruciate ligament (ACL) tear, "
        "or meniscus tear — three binary decisions.")

def process_case(npy_paths: dict, case_id: str, labels: dict, idx: int):
    # npy_paths = {"axial": path, "coronal": path, "sagittal": path}
    first_frames, last_frames = [], []
    for view in ["axial","coronal","sagittal"]:
        if view not in npy_paths: continue
        vol=np.load(str(npy_paths[view]))  # (slices, H, W)
        for z in range(vol.shape[0]):
            gray=window_minmax(vol[z]); rgb=to_rgb(gray); rgb=fit_square(rgb,512)
            # overlay label text on the last frame
            ann=rgb.copy()
            txt=f"{view} | abn:{labels.get('abnormal','?')} acl:{labels.get('acl','?')} men:{labels.get('meniscus','?')}"
            cv2.putText(ann,txt,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,COLORS["yellow"],2)
            first_frames.append(rgb); last_frames.append(ann)
    if not first_frames: return None
    meta={"task":"MRNet knee MRI multi-label (abnormal/ACL/meniscus)","dataset":"MRNet-v1.0",
          "case_id":case_id,"modality":"knee MRI (3 views)","labels":labels,
          "fps":FPS,"frames_per_video":len(first_frames),"case_type":"B_3D_volume_stacked"}
    pick=0
    return write_task(PID,TASK_NAME,idx,first_frames[pick],last_frames[pick],
                      first_frames,last_frames,last_frames,PROMPT,meta,FPS)

def main():
    zip_path=DATA_ROOT/"_extracted"/"M-123_MRNet"/"mrnetkneemris"/"MRNet-v1.0.zip"
    extract_dir=DATA_ROOT/"_extracted"/"M-123_MRNet"/"extracted"
    if not extract_dir.exists() and zip_path.exists():
        print(f"  unzipping {zip_path}...")
        # Iterate members so one corrupt local file header doesn't abort everything.
        # extractall() fails fast on BadZipFile; instead log+skip and keep going.
        extract_dir.mkdir(parents=True, exist_ok=True)
        bad=0; ok=0
        with zipfile.ZipFile(str(zip_path),'r') as z:
            for info in z.infolist():
                try:
                    z.extract(info, str(extract_dir))
                    ok+=1
                except (zipfile.BadZipFile, OSError, EOFError) as e:
                    bad+=1
                    print(f"  [zip-skip] {info.filename}: {e}")
        print(f"  unzip: {ok} ok, {bad} skipped")
        if ok == 0:
            raise RuntimeError(f"zip fully corrupt — 0 members extracted from {zip_path}")
    # Find train-abnormal.csv etc
    root=extract_dir/"MRNet-v1.0"
    if not root.exists():
        # maybe flat
        for sub in extract_dir.iterdir():
            if sub.is_dir() and (sub/"train").exists(): root=sub; break
    # Build labels {case_id: {abnormal, acl, meniscus}}
    labels={}
    for split in ["train","valid"]:
        for task in ["abnormal","acl","meniscus"]:
            csvp=root/f"{split}-{task}.csv"
            if not csvp.exists(): continue
            df=pd.read_csv(str(csvp),header=None,names=["case","label"])
            for _,row in df.iterrows():
                cid=f"{split}-{int(row['case']):04d}"
                labels.setdefault(cid,{})[task]=int(row["label"])
    # Find .npy files per view
    cases={}
    for split in ["train","valid"]:
        for view in ["axial","coronal","sagittal"]:
            vd=root/split/view
            if not vd.exists(): continue
            for npy in sorted(vd.glob("*.npy")):
                cid=f"{split}-{npy.stem}"
                cases.setdefault(cid,{})[view]=npy
    print(f"  {len(cases)} MRNet cases")
    i=0
    for cid,views in sorted(cases.items()):
        lbls=labels.get(cid,{})
        d=process_case(views,cid,lbls,i)
        if d: i+=1; print(f"  wrote {d}")

if __name__=="__main__": main()
