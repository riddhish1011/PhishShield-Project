
import os
import re
import ipaddress
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# =========================================================
# PHISHSHIELD — CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PhishShield | URL Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET_PATH = "urls.csv"
GOOGLE_API_ENDPOINT = (
    "https://safebrowsing.googleapis.com/v5/urls:search"
)

RISK_LEVELS = {
    "Low": (0, 3),
    "Medium": (4, 6),
    "High": (7, 10),
}

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd",
    "cutt.ly", "rb.gy", "shorturl.at", "ow.ly",
    "buff.ly", "rebrand.ly", "soo.gd",
}

SUSPICIOUS_WORDS = {
    "login", "signin", "verify", "verification",
    "account", "password", "credential", "secure",
    "update", "confirm", "wallet", "banking",
    "payment", "invoice", "recover", "unlock",
    "suspended", "gift", "prize", "free",
}

SUSPICIOUS_EXTENSIONS = (
    ".exe", ".scr", ".bat", ".cmd", ".msi",
    ".js", ".vbs", ".ps1", ".apk",
)

SUSPICIOUS_TLDS = {
    "zip", "mov", "click", "work", "country",
    "gq", "tk", "top", "xyz", "support",
}

FEATURE_NAMES = [
    "url_length",
    "host_length",
    "path_length",
    "dot_count",
    "hyphen_count",
    "at_count",
    "digit_count",
    "subdomain_count",
    "has_https",
    "host_is_ip",
    "has_punycode",
    "is_shortener",
    "has_suspicious_word",
    "has_suspicious_extension",
    "has_suspicious_tld",
    "has_port",
    "percent_count",
    "has_userinfo",
    "has_query",
]

# =========================================================
# STYLING — DARK NAVY / CYAN / PURPLE
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                #18254a 0%,
                #0b1020 38%,
                #080d19 100%
            );
        color: #eef4ff;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg, #101a33 0%, #0b1020 100%
        );
        border-right: 1px solid #263655;
    }

    h1, h2, h3 {
        color: #f0f6ff !important;
    }

    .hero {
        padding: 25px;
        border-radius: 20px;
        border: 1px solid #28456a;
        background: linear-gradient(
            120deg, #142542 0%, #20234b 65%, #30204c 100%
        );
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0;
    }

    .hero-subtitle {
        color: #b8c9e8;
        font-size: 15px;
        margin-top: 8px;
    }

    .metric-card {
        background: rgba(18, 30, 54, 0.92);
        border: 1px solid #293c5e;
        border-radius: 15px;
        padding: 17px;
        min-height: 100px;
    }

    .metric-label {
        color: #a9bddb;
        font-size: 13px;
    }

    .metric-value {
        color: #68e1f5;
        font-size: 27px;
        font-weight: 800;
        margin-top: 5px;
    }

    .result-card {
        background: rgba(14, 25, 45, 0.95);
        border: 1px solid #344b70;
        border-radius: 17px;
        padding: 22px;
        margin: 12px 0;
    }

    .small-muted {
        color: #aabbd4;
        font-size: 13px;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(
            90deg, #06b6d4, #7c3aed
        );
        border: none;
        border-radius: 10px;
        color: white;
        font-weight: 700;
        min-height: 44px;
    }

    div.stButton > button {
        border-radius: 9px;
    }

    div[data-testid="stMetric"] {
        background: rgba(18, 30, 54, 0.85);
        border: 1px solid #293c5e;
        padding: 13px;
        border-radius: 13px;
    }

    hr {
        border-color: #283956;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# URL NORMALIZATION / FEATURES
# =========================================================

def normalize_url(raw_url):
    """Normalize and validate a URL without visiting it."""
    value = str(raw_url).strip()

    if not value:
        raise ValueError("Enter a URL to scan.")

    if any(ord(ch) < 32 for ch in value):
        raise ValueError("The URL contains invalid control characters.")

    if len(value) > 4096:
        raise ValueError("The URL is too long to scan.")

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        value = "https://" + value

    parsed = urlparse(value)

    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are supported.")

    if not parsed.hostname:
        raise ValueError("Enter a valid URL containing a hostname.")

    try:
        _ = parsed.port
    except ValueError:
        raise ValueError("The URL contains an invalid port.")

    return value, parsed


def get_host_ip(host):
    """Return True when the hostname is an IP address."""
    if not host:
        return False

    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def extract_features(raw_url):
    """Return explainable numerical features for rules and ML."""
    normalized, parsed = normalize_url(raw_url)

    host = (parsed.hostname or "").lower().rstrip(".")
    path = parsed.path or "/"
    host_parts = [part for part in host.split(".") if part]

    try:
        port = parsed.port
    except ValueError:
        port = None

    host_is_ip = get_host_ip(host)
    tld = host_parts[-1] if host_parts else ""

    try:
        decoded_host = host.encode("ascii").decode("idna").lower()
    except (UnicodeError, UnicodeDecodeError):
        decoded_host = host

    url_lower = normalized.lower()

    suspicious_word_matches = sorted({
        word for word in SUSPICIOUS_WORDS
        if re.search(r"(?<![a-z0-9])" + re.escape(word) +
                     r"(?![a-z0-9])", url_lower)
    })

    # Shorteners are matched exactly, not by substring.
    is_shortener = host in SHORTENERS

    has_userinfo = parsed.username is not None

    features = {
        "url_length": len(normalized),
        "host_length": len(host),
        "path_length": len(path),
        "dot_count": host.count("."),
        "hyphen_count": host.count("-"),
        "at_count": normalized.count("@"),
        "digit_count": sum(c.isdigit() for c in normalized),
        "subdomain_count": max(0, len(host_parts) - 2),
        "has_https": int(parsed.scheme.lower() == "https"),
        "host_is_ip": int(host_is_ip),
        "has_punycode": int("xn--" in host),
        "is_shortener": int(is_shortener),
        "has_suspicious_word": int(bool(suspicious_word_matches)),
        "has_suspicious_extension": int(
            any(path.lower().endswith(ext)
                for ext in SUSPICIOUS_EXTENSIONS)
        ),
        "has_suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "has_port": int(port is not None and port not in {80, 443}),
        "percent_count": normalized.count("%"),
        "has_userinfo": int(has_userinfo),
        "has_query": int(bool(parsed.query)),
    }

    details = {
        "normalized_url": normalized,
        "host": host,
        "decoded_host": decoded_host,
        "path": path,
        "tld": tld,
        "matched_words": suspicious_word_matches,
        "is_shortener": is_shortener,
        "has_userinfo": has_userinfo,
        "host_is_ip": host_is_ip,
    }

    return features, details


# =========================================================
# EXPLAINABLE HEURISTIC RISK SCORE
# =========================================================

def heuristic_scan(raw_url):
    features, details = extract_features(raw_url)

    score = 0
    reasons = []

    def add(points, message):
        nonlocal score
        score += points
        reasons.append(message)

    if features["host_is_ip"]:
        add(3, "The hostname is an IP address rather than a domain name.")

    if features["has_punycode"]:
        add(2, "The hostname uses punycode encoding; inspect for lookalike domains.")

    if details["has_userinfo"]:
        add(3, "The URL contains user-information syntax before the hostname.")

    if features["is_shortener"]:
        add(1, "The URL uses a known link-shortening service, which hides its destination.")

    if features["subdomain_count"] >= 4:
        add(2, "The hostname has an unusually large number of subdomains.")
    elif features["subdomain_count"] >= 2:
        add(1, "The hostname contains multiple subdomains.")

    if features["hyphen_count"] >= 4:
        add(2, "The URL contains many hyphens.")
    elif features["hyphen_count"] >= 2:
        add(1, "The URL contains multiple hyphens.")

    if features["url_length"] > 180:
        add(2, "The URL is unusually long.")
    elif features["url_length"] > 100:
        add(1, "The URL is relatively long.")

    if features["has_suspicious_word"]:
        words = ", ".join(details["matched_words"][:6])
        add(2, f"Potentially sensitive wording appears in the URL: {words}.")

    if features["has_suspicious_extension"]:
        add(3, "The path ends with a potentially risky executable or script extension.")

    if features["has_suspicious_tld"]:
        add(1, "The top-level domain is on a heuristic watchlist; this alone proves nothing.")

    if features["has_port"]:
        add(1, "The URL specifies a non-standard web port.")

    if features["percent_count"] >= 5:
        add(2, "The URL contains extensive percent-encoding.")
    elif features["percent_count"] >= 2:
        add(1, "The URL contains repeated percent-encoding.")

    if features["at_count"] > 0 and not details["has_userinfo"]:
        add(1, "The URL contains an @ character that merits inspection.")

    # HTTPS is useful but does not prove that a site is trustworthy.
    if not features["has_https"]:
        add(1, "The URL does not use HTTPS.")

    # Avoid a very low score when multiple strong indicators appear together.
    score = min(10, score)

    if score <= 3:
        level = "Low"
    elif score <= 6:
        level = "Medium"
    else:
        level = "High"

    if not reasons:
        reasons.append("No listed heuristic warning was triggered.")

    return {
        "score": score,
        "level": level,
        "reasons": reasons,
        "features": features,
        "details": details,
    }


# =========================================================
# OPTIONAL ML MODEL + HELD-OUT EVALUATION
# =========================================================

@st.cache_data(show_spinner=False)
def load_dataset(path, modified_time):
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["url", "label"])

    if not {"url", "label"}.issubset(df.columns):
        return pd.DataFrame(columns=["url", "label"])

    df = df[["url", "label"]].dropna().copy()
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna()
    df = df[df["label"].isin([0, 1])]
    df["label"] = df["label"].astype(int)
    df = df[df["url"] != ""].drop_duplicates(subset=["url"])

    return df.reset_index(drop=True)


def build_feature_row(url):
    features, _ = extract_features(url)
    return [features[name] for name in FEATURE_NAMES]


@st.cache_resource(show_spinner=False)
def train_model(csv_path, modified_time):
    df = load_dataset(csv_path, modified_time)

    result = {
        "model": None,
        "metrics": None,
        "dataset_rows": len(df),
        "message": "",
        "report": None,
        "confusion": None,
    }

    if df.empty:
        result["message"] = "Dataset missing, empty, or incorrectly formatted."
        return result

    class_counts = df["label"].value_counts().to_dict()

    if len(class_counts) < 2 or min(class_counts.values()) < 10:
        result["message"] = (
            "At least 10 unique URLs per class are required to attempt "
            "a meaningful train/test split. More data is recommended."
        )
        return result

    try:
        rows = []
        valid_labels = []

        for _, item in df.iterrows():
            try:
                rows.append(build_feature_row(item["url"]))
                valid_labels.append(int(item["label"]))
            except (ValueError, TypeError):
                continue

        X = pd.DataFrame(rows, columns=FEATURE_NAMES)
        y = pd.Series(valid_labels, dtype=int)

        counts = y.value_counts()
        if len(counts) < 2 or min(counts.values) < 10:
            result["message"] = "Not enough valid examples from both classes."
            return result

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )

        model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            min_samples_leaf=2,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        result["model"] = model
        result["metrics"] = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "test_size": len(y_test),
        }
        result["confusion"] = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        )
        result["report"] = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate (0)", "Phishing (1)"],
            zero_division=0,
            output_dict=True,
        )
        result["message"] = "Model trained and evaluated on a held-out test split."

    except Exception as exc:
        result["message"] = f"Model could not be trained: {exc}"

    return result


def get_google_api_key():
    try:
        key = st.secrets.get("GOOGLE_SAFE_BROWSING_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass

    return os.getenv("GOOGLE_SAFE_BROWSING_API_KEY", "").strip()


def google_safe_browsing_check(url):
    """Return status, threat types and message; never claim API errors are safe."""
    api_key = get_google_api_key()

    if not api_key:
        return {
            "status": "Not configured",
            "threat_types": [],
            "message": "Google Safe Browsing API key is not configured.",
        }

    try:
        response = requests.get(
            GOOGLE_API_ENDPOINT,
            params={"key": api_key, "urls": url},
            timeout=12,
        )

        if response.status_code != 200:
            return {
                "status": "Unavailable",
                "threat_types": [],
                "message": (
                    f"Google API returned HTTP {response.status_code}. "
                    "Check API setup, permissions, quota, and endpoint availability."
                ),
            }

        payload = response.json()
        threats = payload.get("threats", [])

        if threats:
            threat_types = sorted({
                threat_type
                for threat in threats
                for threat_type in threat.get("threatTypes", [])
            })

            return {
                "status": "Known threat",
                "threat_types": threat_types,
                "message": "Google Safe Browsing reported a threat match.",
            }

        return {
            "status": "No known threat found",
            "threat_types": [],
            "message": (
                "No match was returned by the threat list. "
                "This does not guarantee the URL is safe."
            ),
        }

    except requests.Timeout:
        return {
            "status": "Unavailable",
            "threat_types": [],
            "message": "The Google Safe Browsing request timed out.",
        }
    except requests.RequestException:
        return {
            "status": "Unavailable",
            "threat_types": [],
            "message": "Could not connect to Google Safe Browsing.",
        }
    except (ValueError, TypeError):
        return {
            "status": "Unavailable",
            "threat_types": [],
            "message": "The API response could not be interpreted.",
        }


# =========================================================
# SESSION STATE
# =========================================================

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

if "last_scan" not in st.session_state:
    st.session_state.last_scan = None

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 🛡️ PhishShield")
    st.caption("Hybrid phishing URL analyzer")
    st.divider()

    page = st.radio(
        "Navigation",
        ["URL Scanner", "Security Dashboard", "Model Evaluation", "About Project"],
        label_visibility="collapsed",
    )

    st.divider()

    api_configured = bool(get_google_api_key())
    st.markdown("### System status")

    st.write(
        "🟢 Google API key configured"
        if api_configured
        else "🟡 Google API not configured"
    )

    try:
        csv_modified = os.path.getmtime(DATASET_PATH)
    except OSError:
        csv_modified = 0

    ml_result = train_model(DATASET_PATH, csv_modified)

    st.write(
        "🟢 ML model available"
        if ml_result["model"] is not None
        else "🟡 ML model unavailable"
    )

    st.caption("A missing optional service does not stop heuristic scanning.")

# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🛡️ PhishShield</div>
        <div class="hero-subtitle">
            Hybrid phishing detection · Explainable risk analysis ·
            Threat intelligence
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SCAN PROCESS
# =========================================================

def run_scan(raw_url):
    heuristic = heuristic_scan(raw_url)
    normalized_url = heuristic["details"]["normalized_url"]

    google_result = google_safe_browsing_check(normalized_url)

    ml_prediction = None
    ml_probability = None

    model = ml_result.get("model")
    if model is not None:
        try:
            row = pd.DataFrame(
                [build_feature_row(normalized_url)],
                columns=FEATURE_NAMES,
            )
            ml_prediction = int(model.predict(row)[0])

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(row)[0]
                class_index = list(model.classes_).index(1)
                ml_probability = float(probabilities[class_index])
        except Exception:
            ml_prediction = None
            ml_probability = None

    score = heuristic["score"]
    reasons = list(heuristic["reasons"])

    # Known threat intelligence match is a strong independent signal.
    if google_result["status"] == "Known threat":
        score = 10
        reasons.insert(
            0,
            "Google Safe Browsing returned a known-threat match."
        )

    else:
        # ML is supplementary; it cannot override a known threat match.
        # Only add a small score adjustment when the model is available.
        if ml_prediction == 1:
            score = min(10, score + 2)
            reasons.append(
                "The trained ML model classified this URL as phishing."
            )
        elif ml_prediction == 0 and score >= 4:
            reasons.append(
                "The ML model classified this URL as legitimate, "
                "but heuristic warnings remain. Review the evidence."
            )

    if score <= 3:
        level = "Low"
    elif score <= 6:
        level = "Medium"
    else:
        level = "High"

    if google_result["status"] == "Known threat":
        verdict = "Known malicious URL"
    elif level == "High":
        verdict = "High-risk URL"
    elif level == "Medium":
        verdict = "Suspicious URL"
    elif google_result["status"] == "No known threat found":
        verdict = "No known threat found; low heuristic risk"
    else:
        verdict = "Low heuristic risk; external check unavailable"

    result = {
        "url": normalized_url,
        "score": score,
        "level": level,
        "verdict": verdict,
        "reasons": list(dict.fromkeys(reasons)),
        "google": google_result,
        "ml_prediction": ml_prediction,
        "ml_probability": ml_probability,
        "heuristic_score": heuristic["score"],
        "features": heuristic["features"],
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    st.session_state.last_scan = result
    st.session_state.scan_history.insert(0, result)
    st.session_state.scan_history = st.session_state.scan_history[:100]

    return result


# =========================================================
# PAGE: URL SCANNER
# =========================================================

if page == "URL Scanner":
    st.subheader("🔎 Analyze a URL")
    st.write(
        "Enter a URL to inspect its characteristics and check known threats."
    )

    with st.form("scan_form"):
        url_input = st.text_input(
            "URL to scan",
            placeholder="https://example.com/login",
            help="Only the URL string is analyzed. The app does not open the website.",
        )

        submitted = st.form_submit_button(
            "🛡️ Scan URL",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            with st.spinner("Analyzing URL and checking available services..."):
                scan = run_scan(url_input)
        except ValueError as exc:
            st.error(str(exc))
            scan = None
        except Exception as exc:
            st.error(f"Scan could not be completed: {exc}")
            scan = None

    scan = st.session_state.last_scan

    if scan:
        st.divider()

        st.markdown("### Scan result")

        color = {
            "Low": "#34d399",
            "Medium": "#fbbf24",
            "High": "#fb7185",
        }[scan["level"]]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Risk score</div>
                    <div class="metric-value">{scan['score']}/10</div>
                    <div class="small-muted">
                        Heuristic + available evidence
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Risk level</div>
                    <div style="color:{color};font-size:27px;
                                font-weight:800;margin-top:5px">
                        {scan['level']}
                    </div>
                    <div class="small-muted">
                        {RISK_LEVELS[scan['level']][0]}–{RISK_LEVELS[scan['level']][1]}
                        risk range
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Detection outcome</div>
                    <div style="color:#e4edff;font-size:17px;
                                font-weight:700;margin-top:10px">
                        {scan['verdict']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="small-muted">Scanned URL</div>
                <div style="overflow-wrap:anywhere;font-weight:600">
                    {scan['url']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Evidence and warnings")
        for reason in scan["reasons"]:
            st.markdown(f"- {reason}")

        st.markdown("#### Threat intelligence")

        google_status = scan["google"]["status"]
        if google_status == "Known threat":
            st.error(scan["google"]["message"])
            if scan["google"]["threat_types"]:
                st.write(
                    "Threat types:",
                    ", ".join(scan["google"]["threat_types"]),
                )
        elif google_status == "No known threat found":
            st.success(scan["google"]["message"])
        elif google_status == "Not configured":
            st.info(scan["google"]["message"])
        else:
            st.warning(scan["google"]["message"])

        st.markdown("#### Machine learning")

        if scan["ml_prediction"] is None:
            st.info(
                "ML prediction unavailable. The heuristic scan and any "
                "available threat-intelligence result are shown independently."
            )
        else:
            label = (
                "Phishing (1)"
                if scan["ml_prediction"] == 1
                else "Legitimate (0)"
            )
            st.write(f"Model classification: **{label}**")
            if scan["ml_probability"] is not None:
                st.caption(
                    "Model-estimated phishing score: "
                    f"{scan['ml_probability']:.1%}. This is not a calibrated "
                    "probability that the URL is malicious."
                )

        with st.expander("Technical feature details"):
            feature_df = pd.DataFrame(
                [
                    {"Feature": key, "Value": value}
                    for key, value in scan["features"].items()
                ]
            )
            st.dataframe(feature_df, use_container_width=True, hide_index=True)

        st.caption(
            "Important: heuristic risk scores are not probabilities. "
            "A low score or no threat-list match cannot guarantee safety."
        )

    st.divider()
    st.markdown("### Risk guide")

    guide_cols = st.columns(3)
    guide_cols[0].success("**Low: 0–3**\n\nFew heuristic indicators")
    guide_cols[1].warning("**Medium: 4–6**\n\nReview the warning evidence")
    guide_cols[2].error("**High: 7–10**\n\nElevated risk; avoid interacting until verified")


# =========================================================
# PAGE: SECURITY DASHBOARD
# =========================================================

elif page == "Security Dashboard":
    st.subheader("📊 Security Dashboard")

    history = st.session_state.scan_history

    if not history:
        st.info("No scans yet. Go to URL Scanner and analyze a URL first.")
    else:
        df_history = pd.DataFrame(history)

        total = len(df_history)
        high = int((df_history["level"] == "High").sum())
        medium = int((df_history["level"] == "Medium").sum())
        low = int((df_history["level"] == "Low").sum())
        known_threats = int(
            (df_history["google"].apply(
                lambda x: x["status"] == "Known threat"
            )).sum()
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total scans", total)
        c2.metric("High risk", high)
        c3.metric("Medium risk", medium)
        c4.metric("Known threat matches", known_threats)

        st.markdown("### Risk distribution")
        distribution = pd.DataFrame(
            {
                "Risk level": ["Low", "Medium", "High"],
                "Scans": [low, medium, high],
            }
        ).set_index("Risk level")

        st.bar_chart(distribution)

        st.markdown("### Recent scans")
        display_df = pd.DataFrame(
            [
                {
                    "Time": item["timestamp"],
                    "URL": item["url"],
                    "Score": f"{item['score']}/10",
                    "Risk": item["level"],
                    "Verdict": item["verdict"],
                    "Google check": item["google"]["status"],
                }
                for item in history
            ]
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download scan history (CSV)",
            data=csv_bytes,
            file_name="phishshield_scan_history.csv",
            mime="text/csv",
        )

        if st.button("Clear scan history"):
            st.session_state.scan_history = []
            st.session_state.last_scan = None
            st.rerun()


# =========================================================
# PAGE: MODEL EVALUATION
# =========================================================

elif page == "Model Evaluation":
    st.subheader("🧠 Machine Learning Evaluation")
    st.write(
        "The model uses URL-derived features and a held-out test split. "
        "Metrics are displayed only when training and evaluation succeed."
    )

    df = load_dataset(DATASET_PATH, csv_modified)

    st.markdown("### Dataset status")

    d1, d2, d3 = st.columns(3)
    d1.metric("Unique labeled URLs", len(df))
    d2.metric("Legitimate examples", int((df["label"] == 0).sum()))
    d3.metric("Phishing examples", int((df["label"] == 1).sum()))

    if df.empty:
        st.warning(
            "Create urls.csv with columns url,label. Use label 0 for "
            "legitimate and 1 for phishing. Use a verified, representative "
            "dataset and do not label URLs by guessing."
        )
    else:
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    st.markdown("### Evaluation results")

    if ml_result["model"] is None or ml_result["metrics"] is None:
        st.warning(ml_result["message"])
        st.info(
            "A few examples are not enough for meaningful evaluation. "
            "Collect more accurately labeled URLs from legitimate and "
            "phishing sources before relying on the model."
        )
    else:
        metrics = ml_result["metrics"]

        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        b.metric("Precision", f"{metrics['precision']:.1%}")
        c.metric("Recall", f"{metrics['recall']:.1%}")
        d.metric("F1-score", f"{metrics['f1']:.1%}")

        st.caption(
            f"Evaluated on {metrics['test_size']} held-out examples. "
            "These metrics describe this dataset and split, not guaranteed "
            "real-world performance."
        )

        st.markdown("### Confusion matrix")
        cm = ml_result["confusion"]
        cm_df = pd.DataFrame(
            cm,
            index=["Actual legitimate", "Actual phishing"],
            columns=["Predicted legitimate", "Predicted phishing"],
        )
        st.dataframe(cm_df, use_container_width=True)

        st.markdown("### Classification report")
        report = ml_result["report"]
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)


# =========================================================
# PAGE: ABOUT
# =========================================================

elif page == "About Project":
    st.subheader("About PhishShield")

    st.markdown(
        """
        **PhishShield** is a hybrid phishing URL analysis project built
        with Python and Streamlit.

        ### Core components
        - Explainable, rule-based URL risk analysis
        - Google Safe Browsing threat-list lookups
        - Optional Random Forest classifier trained on labeled URL data
        - Held-out evaluation using accuracy, precision, recall, F1-score,
          and a confusion matrix
        - Scan history and risk-distribution dashboard

        ### Dataset format

        `urls.csv` must contain these two columns:

        | url | label |
        |---|---:|
        | `https://example.com` | 0 |
        | `https://verified-test-phishing.example/path` | 1 |

        The sample above illustrates the format only. Use real, correctly
        labeled examples from a trustworthy dataset for model training.

        ### Limitations
        - URL patterns can create false positives and false negatives.
        - Threat intelligence may not know about a newly created phishing URL.
        - An API timeout is not evidence that a URL is safe.
        - HTTPS encrypts a connection but does not prove a site is trustworthy.
        - ML performance depends on data quality and test methodology.

        **PhishShield is a decision-support tool, not a guarantee of safety.**
        """
    )

    st.markdown("### References")
    st.markdown(
        "- [Google Safe Browsing API](https://developers.google.com/safe-browsing/reference/rest)"
    )
    st.markdown(
        "- [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)"
    )

# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption(
    "PhishShield • Educational cybersecurity project • "
    "Always verify suspicious links through trusted channels."
)
