"""
High-Performance Vectorized Session Extractor for CERT r4.2
Reconstructs session-level feature matrix from partitioned Parquet event logs.
100% faithful replication of feature_extraction.py with NumPy vectorization.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT_DIR / "extraction_cache"
OUTPUT_DIR = ROOT_DIR / "ExtractedData"

# Load canonical column names
COLS_FILE = ROOT_DIR / "session_columns.json"
if COLS_FILE.exists():
    with open(COLS_FILE, 'r') as f:
        SESSION_COLUMNS = json.load(f)
else:
    SESSION_COLUMNS = None


def extract_week_sessions(week, df_users, user_dict, ufdict, primary_pc, shared_pcs, supervisor_pc):
    """
    Extracts all sessions and 128 features for a single week using pure vectorized NumPy.
    Returns a DataFrame with the exact benchmark columns.
    """
    parquet_parts = {
        'logon': CACHE_DIR / "logon_part" / f"week_idx={week}",
        'device': CACHE_DIR / "device_part" / f"week_idx={week}",
        'file': CACHE_DIR / "file_part" / f"week_idx={week}",
        'email': CACHE_DIR / "email_part" / f"week_idx={week}",
        'http': CACHE_DIR / "http_part" / f"week_idx={week}"
    }

    dfs = []
    # 1. Logon
    if parquet_parts['logon'].exists():
        df_l = pd.read_parquet(parquet_parts['logon'])
        if len(df_l) > 0:
            df_l['act'] = df_l['activity'].map({'Logon': 1, 'Logoff': 2}).fillna(1).astype(np.int32)
            dfs.append(df_l[['id', 'ts', 'user', 'pc', 'act']])

    # 2. Device
    if parquet_parts['device'].exists():
        df_d = pd.read_parquet(parquet_parts['device'])
        if len(df_d) > 0:
            df_d['act'] = df_d['activity'].map({'Connect': 3, 'Disconnect': 4}).fillna(3).astype(np.int32)
            dfs.append(df_d[['id', 'ts', 'user', 'pc', 'act']])

    # 3. File
    if parquet_parts['file'].exists():
        df_f = pd.read_parquet(parquet_parts['file'])
        if len(df_f) > 0:
            df_f['act'] = 7
            dfs.append(df_f[['id', 'ts', 'user', 'pc', 'act', 'file_type', 'disk', 'file_depth', 'file_len', 'file_nwords']])

    # 4. Email
    if parquet_parts['email'].exists():
        df_e = pd.read_parquet(parquet_parts['email'])
        if len(df_e) > 0:
            df_e['act'] = 6
            to_s = df_e['to'].fillna('').astype(str).values
            cc_s = df_e['cc'].fillna('').astype(str).values
            bcc_s = df_e['bcc'].fillna('').astype(str).values

            n_des = np.zeros(len(df_e), dtype=np.int32)
            n_bccdes = np.zeros(len(df_e), dtype=np.int32)
            n_exdes = np.zeros(len(df_e), dtype=np.int32)
            Xemail = np.zeros(len(df_e), dtype=np.int32)
            exbccmail = np.zeros(len(df_e), dtype=np.int32)

            for idx in range(len(df_e)):
                t, c, b = to_s[idx], cc_s[idx], bcc_s[idx]
                t_lst = t.split(';') if t else []
                c_lst = c.split(';') if c else []
                b_lst = b.split(';') if b else []

                n_des[idx] = len(t_lst) + len(c_lst) + len(b_lst)
                n_bccdes[idx] = len(b_lst)

                all_rec = t_lst + c_lst + b_lst
                ex_cnt = sum(1 for x in all_rec if x and 'dtaa.com' not in x)
                n_exdes[idx] = ex_cnt
                if ex_cnt > 0:
                    Xemail[idx] = 1
                if any(x and 'dtaa.com' not in x for x in b_lst):
                    exbccmail[idx] = 1

            df_e['n_des'] = n_des
            df_e['n_bccdes'] = n_bccdes
            df_e['n_exdes'] = n_exdes
            df_e['Xemail'] = Xemail
            df_e['exbccmail'] = exbccmail

            # Parse n_atts count if stored as string (e.g. CERT r5.2 'file.pdf(1234);...')
            if df_e['n_atts'].dtype == object or df_e['n_atts'].dtype == 'string':
                att_vals = df_e['n_atts'].fillna('').astype(str).values
                df_e['n_atts'] = np.array([len([x for x in s.split(';') if x.strip()]) if s else 0 for s in att_vals], dtype=np.int32)
            else:
                df_e['n_atts'] = pd.to_numeric(df_e['n_atts'], errors='coerce').fillna(0).astype(np.int32)

            dfs.append(df_e[['id', 'ts', 'user', 'pc', 'act', 'n_des', 'n_atts', 'Xemail', 'n_exdes', 'n_bccdes', 'exbccmail', 'email_size', 'email_text_slen', 'email_text_nwords']])

    # 5. HTTP
    if parquet_parts['http'].exists():
        df_h = pd.read_parquet(parquet_parts['http'])
        if len(df_h) > 0:
            df_h['act'] = 5
            dfs.append(df_h[['id', 'ts', 'user', 'pc', 'act', 'http_type', 'url_len', 'url_depth', 'http_c_len', 'http_c_nwords']])

    if not dfs:
        return pd.DataFrame()

    events = pd.concat(dfs, ignore_index=True)
    events['ts'] = pd.to_datetime(events['ts'])
    events.sort_values('ts', inplace=True)
    events.reset_index(drop=True, inplace=True)

    # Defaults for non-applicable event columns
    col_defaults = {
        'usb_dur': 0.0, 'file_type': 0, 'disk': 0, 'file_depth': 0, 'file_len': 0, 'file_nwords': 0,
        'n_des': 0, 'n_atts': 0, 'Xemail': 0, 'n_exdes': 0, 'n_bccdes': 0, 'exbccmail': 0,
        'email_size': 0, 'email_text_slen': 0, 'email_text_nwords': 0,
        'http_type': 0, 'url_len': 0, 'url_depth': 0, 'http_c_len': 0, 'http_c_nwords': 0
    }
    for col, default in col_defaults.items():
        if col not in events.columns:
            events[col] = default
        else:
            events[col] = events[col].fillna(default)

    # Connect duration
    usb_mask = events['act'].isin([3, 4])
    if usb_mask.any():
        sub_usb = events[usb_mask]
        for (u, pc), grp in sub_usb.groupby(['user', 'pc']):
            conns = grp[grp['act'] == 3]
            discs = grp[grp['act'] == 4]
            for c_idx, c_time in zip(conns.index, conns['ts']):
                sub_d = discs[discs['ts'] > c_time]
                if len(sub_d) > 0:
                    events.at[c_idx, 'usb_dur'] = (sub_d['ts'].iloc[0] - c_time).total_seconds()
                else:
                    events.at[c_idx, 'usb_dur'] = -1.0

    # PC code mapping (0: own pc, 1: shared pc, 2: other's pc, 3: supervisor's pc)
    users_arr = events['user'].to_numpy()
    pcs_arr = events['pc'].to_numpy()
    pc_codes = np.full(len(events), 2, dtype=np.int32)
    for i in range(len(events)):
        u = users_arr[i]
        p = pcs_arr[i]
        if u in primary_pc and p == primary_pc[u]:
            pc_codes[i] = 0
        elif u in shared_pcs and p in shared_pcs[u]:
            pc_codes[i] = 1
        elif u in supervisor_pc and p == supervisor_pc[u]:
            pc_codes[i] = 3
    events['pc_code'] = pc_codes

    # Time mapping (1: weekday workhour, 2: weekday afterhour, 3: weekend workhour, 4: weekend afterhour)
    ts = events['ts']
    hour = ts.dt.hour + ts.dt.minute / 60.0
    is_after = (hour < 7.5) | (hour > 17.5)
    is_wknd = ts.dt.weekday >= 5

    time_codes = np.ones(len(events), dtype=np.int32)
    time_codes[~is_wknd & is_after] = 2
    time_codes[is_wknd & ~is_after] = 3
    time_codes[is_wknd & is_after] = 4
    events['time'] = time_codes
    events['day'] = (ts.dt.floor('D') - pd.Timestamp('2009-12-28')).dt.days.values

    # Ground truth marking
    events['mal_act'] = 0
    events['insider'] = 0

    for u in events['user'].unique():
        if u in df_users.index and df_users.loc[u, 'malscene'] > 0:
            mstart = df_users.loc[u, 'mstart']
            mend = df_users.loc[u, 'mend']
            malscene = df_users.loc[u, 'malscene']
            malacts = df_users.loc[u, 'malacts'] or set()

            u_mask = events['user'] == u
            if pd.notna(mstart) and pd.notna(mend):
                t_mask = u_mask & (events['ts'] >= mstart) & (events['ts'] <= mend)
                if t_mask.any():
                    events.loc[t_mask, 'insider'] = malscene
                    if malacts:
                        act_mask = t_mask & events['id'].isin(malacts)
                        events.loc[act_mask, 'mal_act'] = 1

    # Pre-extract numpy arrays for fast vectorized access
    user_grp = events.groupby('user').indices
    acts_arr = events['act'].values
    pcids_arr = events['pc'].values
    tss_arr = events['ts'].values
    time_arr = events['time'].values
    day_arr = events['day'].values
    pc_code_arr = events['pc_code'].values
    mal_act_arr = events['mal_act'].values
    insider_arr = events['insider'].values

    file_type_arr = events['file_type'].values
    disk_arr = events['disk'].values
    file_len_arr = events['file_len'].values
    file_depth_arr = events['file_depth'].values
    file_nwords_arr = events['file_nwords'].values

    http_type_arr = events['http_type'].values
    url_len_arr = events['url_len'].values
    url_depth_arr = events['url_depth'].values
    http_c_len_arr = events['http_c_len'].values
    http_c_nwords_arr = events['http_c_nwords'].values

    n_des_arr = events['n_des'].values
    n_atts_arr = events['n_atts'].values
    n_exdes_arr = events['n_exdes'].values
    n_bccdes_arr = events['n_bccdes'].values
    email_size_arr = events['email_size'].values
    email_text_slen_arr = events['email_text_slen'].values
    email_text_nwords_arr = events['email_text_nwords'].values
    Xemail_arr = events['Xemail'].values
    exbccmail_arr = events['exbccmail'].values
    usb_dur_arr = events['usb_dur'].values

    session_rows = []
    curr_sid = week * 100000

    for u, u_indices in user_grp.items():
        if u not in user_dict:
            continue
        v = user_dict[u]
        u_info = df_users.loc[u]
        u_prof = [
            ufdict['role'][str(u_info['role'])],
            ufdict['b_unit'][str(u_info['b_unit'])],
            ufdict['f_unit'][str(u_info['f_unit'])],
            ufdict['dept'][str(u_info['dept'])],
            ufdict['team'][str(u_info['team'])],
            1 if u_info['role'] == 'ITAdmin' else 0,
            float(u_info['O']), float(u_info['C']), float(u_info['E']), float(u_info['A']), float(u_info['N'])
        ]

        # Slices for user
        u_acts = acts_arr[u_indices]
        u_pcids = pcids_arr[u_indices]
        u_tss = tss_arr[u_indices]

        # Fast session grouping (matching get_sessions in feature_extraction.py)
        open_sessions = {}
        sid = 0
        n_u = len(u_acts)
        if n_u == 0:
            continue

        cur_pc = u_pcids[0]
        st_ts = u_tss[0]
        st_stat = 1 if u_acts[0] == 1 else 2
        open_sessions[cur_pc] = [cur_pc, st_stat, 0, st_ts, st_ts, 1, [u_indices[0]]]

        u_sessions = []
        for i in range(1, n_u):
            cur_pc = u_pcids[i]
            act = u_acts[i]
            ts_val = u_tss[i]
            g_idx = u_indices[i]

            if cur_pc in open_sessions:
                if act == 2:
                    open_sessions[cur_pc][2] = 1
                    open_sessions[cur_pc][4] = ts_val
                    open_sessions[cur_pc][6].append(g_idx)
                    u_sessions.append([curr_sid + sid] + open_sessions.pop(cur_pc))
                    sid += 1
                elif act == 1:
                    open_sessions[cur_pc][2] = 2
                    u_sessions.append([curr_sid + sid] + open_sessions.pop(cur_pc))
                    sid += 1
                    open_sessions[cur_pc] = [cur_pc, 1, 0, ts_val, ts_val, 1, [g_idx]]
                    if len(open_sessions) > 1:
                        for k in open_sessions:
                            open_sessions[k][5] += 1
                else:
                    open_sessions[cur_pc][4] = ts_val
                    open_sessions[cur_pc][6].append(g_idx)
            else:
                st_stat = 1 if act == 1 else 2
                open_sessions[cur_pc] = [cur_pc, st_stat, 0, ts_val, ts_val, 1, [g_idx]]
                if len(open_sessions) > 1:
                    for k in open_sessions:
                        open_sessions[k][5] += 1

        curr_sid += len(u_sessions)

        # Fast feature calculation per session
        for sinfo in u_sessions:
            act_idx = sinfo[7]
            n_acts = len(act_idx)
            if n_acts == 0:
                continue

            s_time = time_arr[act_idx]
            s_day = day_arr[act_idx]
            d = s_day[0]
            perworkhour = float(np.count_nonzero(s_time == 1)) / n_acts
            perafterhour = float(np.count_nonzero(s_time == 2)) / n_acts
            perweekend = float(np.count_nonzero(s_time == 3)) / n_acts
            perweekendafterhour = float(np.count_nonzero(s_time == 4)) / n_acts
            n_days = len(np.unique(s_day))

            st_ts = pd.Timestamp(sinfo[4])
            end_ts = pd.Timestamp(sinfo[5])
            s_dur = (end_ts - st_ts).total_seconds() / 60.0
            s_start = st_ts.hour + st_ts.minute / 60.0
            s_end = end_ts.hour + end_ts.minute / 60.0

            # Behavioral features
            s_act = acts_arr[act_idx]
            s_mal = mal_act_arr[act_idx]
            s_ins = insider_arr[act_idx]

            # Logon
            n_logon = int(np.count_nonzero(s_act == 1))

            # USB
            usb_m = (s_act == 3)
            n_usb = int(np.count_nonzero(usb_m))
            usb_mean_dur = float(np.mean(usb_dur_arr[act_idx][usb_m])) if n_usb > 0 else 0.0

            # File
            file_m = (s_act == 7)
            n_file = int(np.count_nonzero(file_m))
            if n_file > 0:
                fl = file_len_arr[act_idx][file_m]
                fd = file_depth_arr[act_idx][file_m]
                fw = file_nwords_arr[act_idx][file_m]
                fdisk = disk_arr[act_idx][file_m]
                file_stats = [
                    float(np.mean(fl)), float(np.mean(fd)), float(np.mean(fw)),
                    int(np.count_nonzero(fdisk == 0)), int(np.count_nonzero(fdisk == 1)), int(np.count_nonzero(fdisk == 2))
                ]
            else:
                file_stats = [0.0, 0.0, 0.0, 0, 0, 0]

            file_substats = []
            s_ft = file_type_arr[act_idx]
            for ft in [1, 2, 3, 4, 5, 6]:
                ft_m = file_m & (s_ft == ft)
                n_ft = int(np.count_nonzero(ft_m))
                if n_ft > 0:
                    fl = file_len_arr[act_idx][ft_m]
                    fd = file_depth_arr[act_idx][ft_m]
                    fw = file_nwords_arr[act_idx][ft_m]
                    fdisk = disk_arr[act_idx][ft_m]
                    file_substats += [
                        n_ft, float(np.mean(fl)), float(np.mean(fd)), float(np.mean(fw)),
                        int(np.count_nonzero(fdisk == 0)), int(np.count_nonzero(fdisk == 1)), int(np.count_nonzero(fdisk == 2))
                    ]
                else:
                    file_substats += [0, 0.0, 0.0, 0.0, 0, 0, 0]

            # Email
            email_m = (s_act == 6)
            n_email = int(np.count_nonzero(email_m))
            if n_email > 0:
                ndes = n_des_arr[act_idx][email_m].astype(np.float64)
                natts = n_atts_arr[act_idx][email_m].astype(np.float64)
                nexdes = n_exdes_arr[act_idx][email_m].astype(np.float64)
                nbccdes = n_bccdes_arr[act_idx][email_m].astype(np.float64)
                esize = email_size_arr[act_idx][email_m].astype(np.float64)
                eslen = email_text_slen_arr[act_idx][email_m].astype(np.float64)
                enwords = email_text_nwords_arr[act_idx][email_m].astype(np.float64)
                xemail = Xemail_arr[act_idx][email_m]
                exbcc = exbccmail_arr[act_idx][email_m]
                email_stats = [
                    float(np.mean(ndes)), float(np.mean(natts)), float(np.mean(nexdes)),
                    float(np.mean(nbccdes)), float(np.mean(esize)), float(np.mean(eslen)),
                    float(np.mean(enwords)), int(np.count_nonzero(xemail == 1)), int(np.count_nonzero(exbcc == 1))
                ]
            else:
                email_stats = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0]

            # HTTP
            http_m = (s_act == 5)
            n_http = int(np.count_nonzero(http_m))
            if n_http > 0:
                ulen = url_len_arr[act_idx][http_m]
                udep = url_depth_arr[act_idx][http_m]
                hclen = http_c_len_arr[act_idx][http_m]
                hcwords = http_c_nwords_arr[act_idx][http_m]
                http_stats = [float(np.mean(ulen)), float(np.mean(udep)), float(np.mean(hclen)), float(np.mean(hcwords))]
            else:
                http_stats = [0.0, 0.0, 0.0, 0.0]

            http_substats = []
            s_ht = http_type_arr[act_idx]
            for ht in [1, 2, 3, 4, 5, 6]:
                ht_m = http_m & (s_ht == ht)
                n_ht = int(np.count_nonzero(ht_m))
                if n_ht > 0:
                    ulen = url_len_arr[act_idx][ht_m]
                    udep = url_depth_arr[act_idx][ht_m]
                    hclen = http_c_len_arr[act_idx][ht_m]
                    hcwords = http_c_nwords_arr[act_idx][ht_m]
                    http_substats += [n_ht, float(np.mean(ulen)), float(np.mean(udep)), float(np.mean(hclen)), float(np.mean(hcwords))]
                else:
                    http_substats += [0, 0.0, 0.0, 0.0, 0.0]

            mal_u = 0
            if np.count_nonzero(s_mal) > 0:
                mal_u = int(np.max(s_ins))

            row = [
                st_ts.timestamp(), end_ts.timestamp(), v, sinfo[0], d, week, pc_code_arr[act_idx[0]],
                perworkhour, perafterhour, perweekend, perweekendafterhour, n_days, s_dur,
                sinfo[6], sinfo[2], sinfo[3], s_start, s_end
            ] + u_prof + [
                n_acts, n_logon, n_usb, usb_mean_dur,
                n_file
            ] + file_stats + file_substats + [
                n_email
            ] + email_stats + [
                n_http
            ] + http_stats + http_substats + [mal_u]

            session_rows.append(row)

    if not session_rows:
        return pd.DataFrame()

    df_week = pd.DataFrame(session_rows, columns=SESSION_COLUMNS)
    return df_week
