"""
Fast, Memory-Safe Feature Extraction for CERT r5.2 on Windows
Optimized with DuckDB + Pure NumPy Vectorized Session Extractor + Parquet Checkpointing
100% faithful replication of CERT benchmark methodology with NumPy vectorization.
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import duckdb
import pandas as pd
import numpy as np
from tqdm import tqdm

from session_extractor import extract_week_sessions

# --- Configuration & Paths ---
ROOT_DIR = Path(__file__).resolve().parent
R5_DIR = ROOT_DIR / "r5.2"
ANSWERS_DIR = ROOT_DIR / "answers"
CACHE_DIR = ROOT_DIR / "extraction_cache"
OUTPUT_DIR = ROOT_DIR / "ExtractedData"

FIRST_DATE = datetime(2009, 12, 27)  # Sunday week 0 start
SD_MONDAY = datetime(2009, 12, 28)   # Day 0 reference

def setup_directories():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 1. DuckDB Partitioning of Raw Event Logs
# ----------------------------------------------------------------------
def partition_raw_logs():
    """
    Streams and partitions all 5 raw event logs into weekly Parquet partitions in CACHE_DIR.
    Runs once; cached outputs are reused across runs.
    """
    csv_opts = "auto_detect=true, timestampformat='%m/%d/%Y %H:%M:%S'"
    
    # 1. Device
    device_out = CACHE_DIR / "device_part"
    if not device_out.exists():
        print("Partitioning device.csv...")
        t0 = time.time()
        q = f"""
        COPY (
            SELECT id, date as ts, user, pc, activity,
                   (CAST(date AS DATE) - DATE '2009-12-27') // 7 as week_idx
            FROM read_csv('{R5_DIR.as_posix()}/device.csv', {csv_opts})
        ) TO '{device_out.as_posix()}' (FORMAT PARQUET, PARTITION_BY (week_idx), OVERWRITE_OR_IGNORE true)
        """
        duckdb.query(q)
        print(f"  Device partitioned in {time.time()-t0:.2f}s")
    else:
        print("  device_part already cached.")
        
    # 2. Logon
    logon_out = CACHE_DIR / "logon_part"
    if not logon_out.exists():
        print("Partitioning logon.csv...")
        t0 = time.time()
        q = f"""
        COPY (
            SELECT id, date as ts, user, pc, activity,
                   (CAST(date AS DATE) - DATE '2009-12-27') // 7 as week_idx
            FROM read_csv('{R5_DIR.as_posix()}/logon.csv', {csv_opts})
        ) TO '{logon_out.as_posix()}' (FORMAT PARQUET, PARTITION_BY (week_idx), OVERWRITE_OR_IGNORE true)
        """
        duckdb.query(q)
        print(f"  Logon partitioned in {time.time()-t0:.2f}s")
    else:
        print("  logon_part already cached.")
        
    # 3. File
    file_out = CACHE_DIR / "file_part"
    if not file_out.exists():
        print("Partitioning file.csv...")
        t0 = time.time()
        q = f"""
        COPY (
            SELECT 
                id, 
                date as ts, 
                user, 
                pc, 
                CASE
                    WHEN filename LIKE '%.zip' OR filename LIKE '%.rar' OR filename LIKE '%.7z' THEN 2
                    WHEN filename LIKE '%.jpg' OR filename LIKE '%.png' OR filename LIKE '%.bmp' THEN 3
                    WHEN filename LIKE '%.doc' OR filename LIKE '%.docx' OR filename LIKE '%.pdf' THEN 4
                    WHEN filename LIKE '%.txt' OR filename LIKE '%.cfg' OR filename LIKE '%.rtf' THEN 5
                    WHEN filename LIKE '%.exe' OR filename LIKE '%.sh' THEN 6
                    ELSE 1
                END AS file_type,
                CASE
                    WHEN filename LIKE 'C%' THEN 1
                    WHEN filename LIKE 'R%' THEN 2
                    ELSE 0
                END AS disk,
                length(filename) - length(replace(filename, '\\\\', '')) AS file_depth,
                length(content) AS file_len,
                length(content) - length(replace(content, ' ', '')) + 1 AS file_nwords,
                (CAST(date AS DATE) - DATE '2009-12-27') // 7 as week_idx
            FROM read_csv('{R5_DIR.as_posix()}/file.csv', {csv_opts})
        ) TO '{file_out.as_posix()}' (FORMAT PARQUET, PARTITION_BY (week_idx), OVERWRITE_OR_IGNORE true)
        """
        duckdb.query(q)
        print(f"  File partitioned in {time.time()-t0:.2f}s")
    else:
        print("  file_part already cached.")
        
    # 4. Email
    email_out = CACHE_DIR / "email_part"
    if not email_out.exists():
        print("Partitioning email.csv (9.42 GB)...")
        t0 = time.time()
        q = f"""
        COPY (
            SELECT 
                id, 
                date as ts, 
                user, 
                pc,
                "to",
                cc,
                bcc,
                size as email_size, 
                CASE 
                    WHEN attachments IS NULL OR attachments = '' THEN 0
                    ELSE length(attachments) - length(replace(attachments, ';', '')) + 1
                END as n_atts,
                length(content) as email_text_slen,
                length(content) - length(replace(content, ' ', '')) + 1 as email_text_nwords,
                (CAST(date AS DATE) - DATE '2009-12-27') // 7 as week_idx
            FROM read_csv('{R5_DIR.as_posix()}/email.csv', {csv_opts})
        ) TO '{email_out.as_posix()}' (FORMAT PARQUET, PARTITION_BY (week_idx), OVERWRITE_OR_IGNORE true)
        """
        duckdb.query(q)
        print(f"  Email partitioned in {time.time()-t0:.2f}s")
    else:
        print("  email_part already cached.")

    # 5. HTTP (30.15 GB) - Single-pass streaming projection into lightweight Parquet
    http_out = CACHE_DIR / "http_part"
    if not http_out.exists():
        print("Partitioning http.csv (30.15 GB) - Streaming C++ projection...")
        t0 = time.time()
        q = f"""
        COPY (
            SELECT 
                id, 
                date as ts, 
                user, 
                pc,
                CASE 
                    WHEN regexp_extract(url, '//([^/]+)/', 1) IN ('dropbox.com', 'drive.google.com', 'mega.co.nz', 'account.live.com') 
                      OR replace(regexp_extract(url, '//([^/]+)/', 1), 'www.', '') IN ('dropbox.com', 'drive.google.com', 'mega.co.nz', 'account.live.com') THEN 3
                    WHEN regexp_extract(url, '//([^/]+)/', 1) IN ('wikileaks.org', 'freedom.press', 'theintercept.com') 
                      OR replace(regexp_extract(url, '//([^/]+)/', 1), 'www.', '') IN ('wikileaks.org', 'freedom.press', 'theintercept.com') THEN 5
                    WHEN regexp_extract(url, '//([^/]+)/', 1) IN ('facebook.com','twitter.com','plus.google.com','instagr.am','instagram.com','flickr.com','linkedin.com','reddit.com','about.com','youtube.com','pinterest.com','tumblr.com','quora.com','vine.co','match.com','t.co')
                      OR replace(regexp_extract(url, '//([^/]+)/', 1), 'www.', '') IN ('facebook.com','twitter.com','plus.google.com','instagr.am','instagram.com','flickr.com','linkedin.com','reddit.com','about.com','youtube.com','pinterest.com','tumblr.com','quora.com','vine.co','match.com','t.co') THEN 2
                    WHEN regexp_extract(url, '//([^/]+)/', 1) IN ('indeed.com','monster.com', 'careerbuilder.com','simplyhired.com')
                      OR replace(regexp_extract(url, '//([^/]+)/', 1), 'www.', '') IN ('indeed.com','monster.com', 'careerbuilder.com','simplyhired.com') THEN 4
                    WHEN (contains(regexp_extract(url, '//([^/]+)/', 1), 'job') AND (contains(regexp_extract(url, '//([^/]+)/', 1), 'hunt') OR contains(regexp_extract(url, '//([^/]+)/', 1), 'search')))
                      OR (contains(regexp_extract(url, '//([^/]+)/', 1), 'aol.com') AND (contains(url, 'recruit') OR contains(url, 'job'))) THEN 4
                    WHEN regexp_extract(url, '//([^/]+)/', 1) IN ('webwatchernow.com','actionalert.com', 'relytec.com','refog.com','wellresearchedreviews.com','softactivity.com', 'spectorsoft.com','best-spy-soft.com')
                      OR replace(regexp_extract(url, '//([^/]+)/', 1), 'www.', '') IN ('webwatchernow.com','actionalert.com', 'relytec.com','refog.com','wellresearchedreviews.com','softactivity.com', 'spectorsoft.com','best-spy-soft.com') THEN 6
                    WHEN contains(regexp_extract(url, '//([^/]+)/', 1), 'keylog') THEN 6
                    ELSE 1
                END AS http_type,
                length(url) as url_len,
                length(url) - length(replace(url, '/', '')) - 2 as url_depth,
                length(content) as http_c_len,
                length(content) - length(replace(content, ' ', '')) + 1 as http_c_nwords,
                (CAST(date AS DATE) - DATE '2009-12-27') // 7 as week_idx
            FROM read_csv('{R5_DIR.as_posix()}/http.csv', {csv_opts})
        ) TO '{http_out.as_posix()}' (FORMAT PARQUET, PARTITION_BY (week_idx), OVERWRITE_OR_IGNORE true)
        """
        duckdb.query(q)
        print(f"  HTTP partitioned in {time.time()-t0:.2f}s")
    else:
        print("  http_part already cached.")

# ----------------------------------------------------------------------
# 2. User & Organizational Hierarchy Setup
# ----------------------------------------------------------------------
def load_user_directory():
    """Parses 18 monthly LDAP snapshots and psychometric scores for r5.2 (2,000 users)."""
    print("Loading LDAP and Psychometric profiles for r5.2...")
    ldap_dir = R5_DIR / "LDAP"
    ldap_files = sorted(list(ldap_dir.glob("*.csv")))
    
    alluser = {}
    alreadyFired = []
    
    for file in ldap_files:
        df_ldap = pd.read_csv(file)
        # r5.2 columns: employee_name, user_id, email, role, projects, business_unit, functional_unit, department, team, supervisor
        employeesThisMonth = []
        for row in df_ldap.itertuples(index=False):
            uid = row[1]
            employeesThisMonth.append(uid)
            if uid not in alluser:
                # Store [uname, email, role, b_unit, f_unit, dept, team, sup, wstart, wend]
                alluser[uid] = [
                    row[0], # employee_name
                    row[2], # email
                    row[3], # role
                    row[5], # business_unit
                    row[6], # functional_unit
                    row[7], # department
                    row[8], # team
                    row[9], # supervisor
                    file.stem,
                    np.nan
                ]
        fired = list(set(alluser.keys()) - set(alreadyFired) - set(employeesThisMonth))
        alreadyFired += fired
        for e in fired:
            alluser[e][-1] = file.stem

    # Psychometric
    psycho_file = R5_DIR / "psychometric.csv"
    p_score = pd.read_csv(psycho_file).values
    for i in range(len(p_score)):
        uid = p_score[i, 1]
        if uid in alluser:
            alluser[uid] = alluser[uid] + list(p_score[i, 2:])

    df_users = pd.DataFrame.from_dict(alluser, orient='index')
    df_users.columns = ['uname', 'email', 'role', 'b_unit', 'f_unit', 'dept', 'team', 'sup', 'wstart', 'wend', 'O', 'C', 'E', 'A', 'N']

    # Map supervisor
    for i in df_users.index:
        sup_name = df_users.loc[i, 'sup']
        if isinstance(sup_name, str):
            sup_match = df_users[df_users['uname'] == sup_name]
            df_users.at[i, 'sup'] = sup_match.index[0] if len(sup_match) > 0 else None
        else:
            df_users.at[i, 'sup'] = None

    print(f"Total users loaded: {len(df_users)}")
    return df_users

def map_user_pcs(df_users):
    """Identifies dedicated and shared PCs using logon and device logs in Weeks 0 and 1."""
    print("Mapping primary and shared workstations...")
    w1_end = (FIRST_DATE + timedelta(weeks=2)).strftime('%Y-%m-%d %H:%M:%S')
    
    csv_opts = "auto_detect=true, timestampformat='%m/%d/%Y %H:%M:%S'"
    q = f"""
    WITH combined_pcs AS (
        SELECT user, pc, (CAST(date AS DATE) - DATE '2009-12-27') // 7 as week_idx
        FROM read_csv('{R5_DIR.as_posix()}/logon.csv', {csv_opts})
        WHERE date < TIMESTAMP '{w1_end}'
        UNION
        SELECT user, pc, (CAST(date AS DATE) - DATE '2009-12-27') // 7 as week_idx
        FROM read_csv('{R5_DIR.as_posix()}/device.csv', {csv_opts})
        WHERE date < TIMESTAMP '{w1_end}'
    )
    SELECT user, week_idx, list(DISTINCT pc) as pcs
    FROM combined_pcs
    GROUP BY user, week_idx
    """
    df_pcs = duckdb.query(q).df()
    
    user_pc_dict = pd.DataFrame(index=df_users.index)
    user_pc_dict['pcs'] = None
    
    w0_map = df_pcs[df_pcs['week_idx'] == 0].set_index('user')['pcs'].to_dict()
    w1_map = df_pcs[df_pcs['week_idx'] == 1].set_index('user')['pcs'].to_dict()
    
    for u in df_users.index:
        pcs0 = set(w0_map.get(u, []))
        pcs1 = set(w1_map.get(u, []))
        intersect = list(pcs0 & pcs1)
        if not intersect:
            intersect = list(pcs0 | pcs1)  # Fallback if no overlap
        user_pc_dict.at[u, 'pcs'] = intersect
        
    user_pc_dict['sharedpc'] = None
    user_pc_dict['npc'] = user_pc_dict['pcs'].apply(lambda x: len(x) if x else 0)
    user_pc_dict.loc[user_pc_dict['npc'] == 1, 'pc'] = user_pc_dict[user_pc_dict['npc'] == 1]['pcs'].apply(lambda x: x[0] if x else None)
    
    multi_pcs = []
    for pcs in user_pc_dict[user_pc_dict['npc'] > 1]['pcs'].values:
        multi_pcs.extend(pcs)
    counts = pd.Series(multi_pcs).value_counts().to_dict() if multi_pcs else {}
    
    for u in user_pc_dict[user_pc_dict['npc'] > 1].index:
        pcs = list(user_pc_dict.loc[u, 'pcs'])
        best_pc = min(pcs, key=lambda p: counts.get(p, 0))
        user_pc_dict.at[u, 'pc'] = best_pc
        if df_users.loc[u, 'role'] != 'ITAdmin':
            shared = [p for p in pcs if p != best_pc]
            user_pc_dict.at[u, 'sharedpc'] = shared
            
    df_users['pc'] = user_pc_dict['pc']
    df_users['sharedpc'] = user_pc_dict['sharedpc']
    print(f"Total workstations mapped: {df_users['pc'].notna().sum()}/{len(df_users)}")
    return df_users

def load_ground_truth(df_users):
    """Loads malicious insiders and exact action IDs from answers/ for r5.2 (all 4 scenarios)."""
    print("Loading ground-truth malicious threat scenarios (Scenarios 1-4)...")
    insiders_file = ANSWERS_DIR / "insiders.csv"
    listmal = pd.read_csv(insiders_file)
    listmal = listmal[listmal['dataset'].astype(str).isin(['5.2', 'r5.2'])].copy()
    
    listmal['start'] = pd.to_datetime(listmal['start'], format='%m/%d/%Y %H:%M:%S')
    listmal['end'] = pd.to_datetime(listmal['end'], format='%m/%d/%Y %H:%M:%S')
    
    df_users['malscene'] = 0
    df_users['mstart'] = pd.NaT
    df_users['mend'] = pd.NaT
    df_users['malacts'] = None
    
    total_actions = 0
    for _, row in listmal.iterrows():
        u = row['user']
        if u in df_users.index:
            df_users.loc[u, 'mstart'] = row['start']
            df_users.loc[u, 'mend'] = row['end']
            df_users.loc[u, 'malscene'] = int(row['scenario'])
            
            detail_file = ANSWERS_DIR / f"r5.2-{row['scenario']}" / row['details']
            if detail_file.exists():
                with open(detail_file, 'r') as f:
                    lines = [line.strip().split(',') for line in f if line.strip()]
                mal_ids = set(x[1].strip('"') for x in lines if len(x) > 3 and x[3].strip('"') == u)
                df_users.at[u, 'malacts'] = mal_ids
                total_actions += len(mal_ids)
                
    print(f"Total malicious employees mapped: {(df_users['malscene'] > 0).sum()} / {len(listmal)}")
    print(f"Total malicious ground-truth actions indexed: {total_actions:,}")
    return df_users

# ----------------------------------------------------------------------
# 3. Main Extraction Loop (Weeks 0 to 74)
# ----------------------------------------------------------------------
def run_session_extraction(df_users, start_week=0, end_week=74):
    """Iterates through weeks 0..74, extracting sessions with Parquet checkpointing."""
    user_dict = {idx: i for i, idx in enumerate(df_users.index)}

    ufdict = {}
    list_uf = ['role', 'b_unit', 'f_unit', 'dept', 'team']
    for f in list_uf:
        df_users[f] = [str(x) for x in df_users[f]]
        tmp = sorted(list(set(df_users[f])))
        ufdict[f] = {idx: i for i, idx in enumerate(tmp)}

    primary_pc = df_users['pc'].to_dict()
    shared_pcs = {u: set(pcs) for u, pcs in df_users['sharedpc'].items() if pcs}
    supervisor_pc = {
        u: df_users.loc[df_users.loc[u, 'sup'], 'pc']
        for u in df_users.index
        if df_users.loc[u, 'sup'] and df_users.loc[u, 'sup'] in df_users.index
    }

    print(f"\nExtracting sessions across Weeks {start_week} to {end_week}...")
    t_start = time.time()
    
    pbar = tqdm(range(start_week, end_week + 1), desc="Extracting Weeks", unit="wk")
    for week in pbar:
        ckpt_file = OUTPUT_DIR / f"sessions_week_{week}.parquet"
        if ckpt_file.exists():
            pbar.set_postfix_str(f"Wk {week} cached")
            continue

        t_w = time.time()
        df_week = extract_week_sessions(
            week, df_users, user_dict, ufdict, primary_pc, shared_pcs, supervisor_pc
        )
        if len(df_week) > 0:
            df_week.to_parquet(ckpt_file, index=False)
            pbar.set_postfix_str(f"Wk {week}: {len(df_week)} ses ({time.time()-t_w:.1f}s)")
        else:
            pbar.set_postfix_str(f"Wk {week}: 0 ses")

    print(f"\nAll weekly sessions extracted in {(time.time() - t_start) / 60:.2f} minutes!")

def assemble_final_dataset(start_week=0, end_week=74):
    """Concatenates all weekly checkpoints into session.parquet and session.csv."""
    print("\nAssembling final dataset from weekly checkpoints...")
    dfs = []
    for week in range(start_week, end_week + 1):
        f = OUTPUT_DIR / f"sessions_week_{week}.parquet"
        if f.exists():
            dfs.append(pd.read_parquet(f))
            
    if not dfs:
        raise RuntimeError("No weekly session files found to assemble!")

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total sessions extracted: {len(full_df):,}")
    print(f"Total features per session: {full_df.shape[1]}")

    # Save session.parquet (fast binary)
    parquet_path = ROOT_DIR / "session.parquet"
    full_df.to_parquet(parquet_path, index=False)
    print(f"Saved: {parquet_path.name} ({parquet_path.stat().st_size / (1024*1024):.2f} MB)")

    # Save session.csv (with default integer index -> Unnamed: 0, 129 columns total)
    csv_path = ROOT_DIR / "session.csv"
    print(f"Writing {csv_path.name}...")
    full_df.to_csv(csv_path, index=True)
    print(f"Saved: {csv_path.name} ({csv_path.stat().st_size / (1024*1024):.2f} MB)")

    # Ground-truth summary
    print("\n" + "="*50)
    print("DATASET VERIFICATION & CLASS BALANCE SUMMARY (CERT r5.2)")
    print("="*50)
    print(f"Total Sessions:   {len(full_df):,}")
    print(f"Columns (Parquet): {full_df.shape[1]}")
    print(f"Columns (CSV):     {full_df.shape[1] + 1} (including 'Unnamed: 0')")
    
    insider_col = full_df['insider']
    benign_count = (insider_col == 0).sum()
    insider_count = (insider_col > 0).sum()
    insider_pct = (insider_count / len(full_df)) * 100
    
    print(f"Benign Sessions:  {benign_count:,} ({100 - insider_pct:.4f}%)")
    print(f"Insider Sessions: {insider_count:,} ({insider_pct:.4f}%)")
    print(f"Scenario Breakdown:")
    for sc in sorted(insider_col.unique()):
        count = (insider_col == sc).sum()
        label = "Benign (Normal)" if sc == 0 else f"Scenario {int(sc)}"
        print(f"  - {label:<20}: {count:,}")
    print("="*50)

if __name__ == "__main__":
    setup_directories()
    partition_raw_logs()
    df_users = load_user_directory()
    df_users = map_user_pcs(df_users)
    df_users = load_ground_truth(df_users)
    run_session_extraction(df_users, start_week=0, end_week=74)
    assemble_final_dataset(start_week=0, end_week=74)
