# NSL-KDD column definitions (KDDTrain+.txt / KDDTest+.txt)
COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

CATEGORICAL = ["protocol_type", "service", "flag"]
LABEL_COL = "label"
DIFFICULTY_COL = "difficulty"

# Map fine-grained attack names -> families (used in multiclass)
FAMILY_MAP = {
    # DoS
    "neptune":"DoS","back":"DoS","land":"DoS","pod":"DoS","smurf":"DoS","teardrop":"DoS",
    # Probe
    "satan":"Probe","ipsweep":"Probe","nmap":"Probe","portsweep":"Probe",
    # R2L
    "guess_passwd":"R2L","ftp_write":"R2L","imap":"R2L","phf":"R2L","multihop":"R2L",
    "warezmaster":"R2L","warezclient":"R2L","spy":"R2L","sendmail":"R2L",
    # U2R
    "buffer_overflow":"U2R","loadmodule":"U2R","perl":"U2R","rootkit":"U2R",
}
