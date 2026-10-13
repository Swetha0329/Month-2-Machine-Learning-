"""MailGuard AI - Flask backend for the Email & SMS spam classifier.

Run with:  python app.py
Needs in the same folder: svm_spam_classifier.joblib, spam.csv, templates/index.html
Files created automatically: history.db (scan history), feedback.csv (user feedback)
"""

import csv
import io
import math
import re
import sqlite3
import threading
import time
import uuid
from collections import Counter, OrderedDict
from datetime import datetime
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr
from html import unescape
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, redirect, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "svm_spam_classifier.joblib"
DATASET_PATH = BASE_DIR / "spam.csv"
DB_PATH = BASE_DIR / "history.db"
FEEDBACK_PATH = BASE_DIR / "feedback.csv"

MAX_MESSAGE_LENGTH = 10000
MAX_BATCH_ROWS = 2000
BATCH_PREVIEW_ROWS = 100
HISTORY_LIMIT = 200
API_BATCH_LIMIT = 50
MAX_STORED_SCANS = 5000
MAX_STORED_BATCHES = 8

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB uploads

model = joblib.load(MODEL_PATH)

# ---------------------------------------------------------------------------
# Labels, prediction and confidence
# ---------------------------------------------------------------------------

HAM_LABELS = {"ham", "not spam", "not_spam", "legitimate", "0", "false"}
SPAM_LABELS = {"spam", "1", "true"}


def is_spam_label(label):
    text = str(label).strip().lower()
    if text in HAM_LABELS:
        return False
    return text in SPAM_LABELS or "spam" in text


def classify_many(texts):
    """Return a list of (is_spam, confidence_percent) for each text."""
    raw = model.predict(texts)
    classes = list(getattr(model, "classes_", []))
    probabilities = None
    scores = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(texts)
    elif hasattr(model, "decision_function") and len(classes) == 2:
        scores = np.asarray(model.decision_function(texts)).reshape(-1)

    results = []
    for index, label in enumerate(raw):
        confidence = 0.0

        if probabilities is not None:
            confidence = float(probabilities[index][classes.index(label)]) * 100
        elif scores is not None:
            confidence = (1 / (1 + math.exp(-min(abs(float(scores[index])), 700)))) * 100

        results.append((is_spam_label(label), max(0.0, min(100.0, confidence))))

    return results


# ---------------------------------------------------------------------------
# Suspicious-word highlighting
# ---------------------------------------------------------------------------

# Everyday words (now, buy, deal, money, gift...) are left out on purpose:
# they appear in normal messages and would cause false highlights.
SPAM_WORDS = {
    "free", "winner", "won", "win", "prize", "claim", "reward",
    "urgent", "congratulations", "congrats", "offer", "limited",
    "cash", "bonus", "lottery", "selected", "guaranteed", "click",
    "clicking", "subscribe", "unsubscribe", "discount", "loan",
    "profit", "investment", "earn", "earnings", "income",
    "promotion", "cheap", "immediately", "verify", "blocked",
    "refund", "risk-free", "txt"
}

SPAM_PATTERN = re.compile(
    r"(?<![\w-])(?:" + "|".join(
        re.escape(word)
        for word in sorted(SPAM_WORDS, key=len, reverse=True)
    ) + r")(?![\w-])",
    re.IGNORECASE
)


def analyze_spam_words(message):
    matches = list(SPAM_PATTERN.finditer(message))
    counts = Counter(match.group(0).lower() for match in matches)

    parts = []
    last = 0

    for match in matches:
        if match.start() > last:
            parts.append({"text": message[last:match.start()], "suspicious": False})
        parts.append({"text": match.group(0), "suspicious": True})
        last = match.end()

    if last < len(message):
        parts.append({"text": message[last:], "suspicious": False})

    frequency = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return parts, frequency


# ---------------------------------------------------------------------------
# Link inspector
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(
    r"""(?ix)
    (?:https?://|www\.)[^\s<>"'\])}]+
    |
    \b(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|ow\.ly|cutt\.ly|rb\.gy|
        shorturl\.at|tiny\.cc)/[^\s<>"'\])}]+
    """
)

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly",
    "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc", "buff.ly", "rebrand.ly"
}

SUSPICIOUS_TLDS = {
    "xyz", "top", "click", "tk", "ml", "ga", "cf", "gq", "work", "zip",
    "mov", "country", "loan", "win", "bid", "stream", "gdn", "icu",
    "cyou", "buzz", "rest", "monster", "support", "download", "men"
}

BRANDS = {
    "paypal": ["paypal.com"],
    "amazon": ["amazon.com", "amazon.in", "amazon.co.uk", "amazonaws.com"],
    "google": ["google.com", "google.co.in", "gmail.com"],
    "microsoft": ["microsoft.com", "live.com", "office.com", "outlook.com"],
    "apple": ["apple.com", "icloud.com"],
    "netflix": ["netflix.com"],
    "facebook": ["facebook.com", "fb.com"],
    "instagram": ["instagram.com"],
    "whatsapp": ["whatsapp.com"],
    "paytm": ["paytm.com"],
    "sbi": ["sbi.co.in", "onlinesbi.com"],
    "hdfc": ["hdfcbank.com"],
    "icici": ["icicibank.com"],
}

LOOKALIKE_MAP = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "$": "s"})
IP_PATTERN = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


def domain_matches(host, domain):
    return host == domain or host.endswith("." + domain)


def lookalike_brand(host):
    """Return the brand a host imitates (e.g. paypa1-login.xyz -> paypal), else None."""
    normalized = host.translate(LOOKALIKE_MAP)
    tokens = re.split(r"[.\-]", normalized)

    for brand, official in BRANDS.items():
        if any(domain_matches(host, domain) for domain in official):
            continue
        for token in tokens:
            if token == brand or (len(brand) >= 5 and brand in token):
                return brand
    return None


def inspect_domain(host):
    """Return (flags, points) for a bare domain name."""
    flags = []
    points = 0

    if IP_PATTERN.match(host):
        flags.append("Uses a raw IP address instead of a domain")
        points += 3

    if host in SHORTENERS:
        flags.append("URL shortener hides the real destination")
        points += 2

    if "xn--" in host:
        flags.append("Punycode domain (can disguise look-alike characters)")
        points += 3

    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    if tld in SUSPICIOUS_TLDS:
        flags.append("Suspicious domain ending (." + tld + ")")
        points += 2

    brand = lookalike_brand(host)
    if brand:
        flags.append("Imitates " + brand.capitalize() + " but is not their official domain")
        points += 4

    if host.count("-") >= 2:
        flags.append("Many hyphens in the domain name")
        points += 1

    if host.count(".") >= 4:
        flags.append("Unusually many subdomains")
        points += 1

    return flags, points


def inspect_links(message):
    rows = []
    seen = set()

    for match in URL_PATTERN.finditer(message):
        url = match.group(0).rstrip(".,;:!?)]}'\"")
        if not url or url.lower() in seen:
            continue
        seen.add(url.lower())

        has_scheme = url.lower().startswith(("http://", "https://"))
        parsed = urlparse(url if has_scheme else "http://" + url)
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]

        flags, points = inspect_domain(host)

        if url.lower().startswith("http://"):
            flags.append("Not encrypted (HTTP instead of HTTPS)")
            points += 1

        if "@" in parsed.netloc:
            flags.append("Contains an @ sign that can hide the real destination")
            points += 3

        if len(url) > 90:
            flags.append("Very long URL")
            points += 1

        if points >= 4:
            level_key, level = "danger", "Dangerous"
        elif points >= 1:
            level_key, level = "caution", "Caution"
        else:
            level_key, level = "safe", "No warning signs"

        rows.append({
            "url": url,
            "host": host or "unknown host",
            "flags": flags,
            "level_key": level_key,
            "level": level,
        })

    return {
        "rows": rows,
        "total": len(rows),
        "dangerous": sum(1 for row in rows if row["level_key"] == "danger"),
        "caution": sum(1 for row in rows if row["level_key"] == "caution"),
    }


# ---------------------------------------------------------------------------
# Smart Threat Radar (rule based, independent of the ML model)
# ---------------------------------------------------------------------------

def make_rules(terms):
    return [
        (term, re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE))
        for term in terms
    ]


WORD_CATEGORIES = [
    {
        "name": "Urgent wording", "cap": 20, "each": 5,
        "rules": make_rules([
            "urgent", "immediately", "act now", "act fast", "right away", "asap",
            "expires", "expire", "expiring", "last chance", "final notice",
            "within 24 hours", "hurry", "limited time", "today only", "deadline",
            "don't miss", "click here", "claim now", "claim your",
        ]),
    },
    {
        "name": "Phishing terms", "cap": 25, "each": 5,
        "rules": make_rules([
            "verify", "password", "suspended", "locked", "blocked", "confirm your",
            "log in", "login", "sign in", "security alert", "unusual activity",
            "otp", "pin", "cvv", "kyc", "update your", "bank account",
            "credit card", "social security", "refund", "permanent closure",
        ]),
    },
    {
        "name": "Money and prize bait", "cap": 25, "each": 5,
        "rules": make_rules([
            "free", "winner", "won", "prize", "lottery", "cash", "reward", "claim",
            "congratulations", "gift card", "bonus", "million", "earn", "income",
            "investment", "loan", "inheritance", "guaranteed", "risk-free",
            "bitcoin", "crypto",
        ]),
    },
]

CURRENCY_PATTERN = re.compile(r"[$£€₹]\s?\d")
EXCLAMATION_PATTERN = re.compile(r"!{2,}")
CAPS_PATTERN = re.compile(r"\b[A-Z]{3,}\b")
CONTACT_PATTERN = re.compile(
    r"\b(?:call|text|txt|reply|whatsapp)\b[^.\n]{0,30}\b(?:now|stop|immediately)\b",
    re.IGNORECASE
)

# The radar chart starts every axis at this value instead of 0. With true zeros, a message
# that triggers only one or two categories collapses into thin spikes with no visible shape.
# The exact points for each category are always shown in the list next to the chart.
RADAR_FLOOR = 12

LEVELS = [
    (25, "low", "Low"),
    (50, "medium", "Medium"),
    (75, "high", "High"),
    (101, "critical", "Critical"),
]


def threat_level(score):
    for limit, key, label in LEVELS:
        if score < limit:
            return key, label
    return "critical", "Critical"


def analyze_threat(message, links):
    categories = []

    for spec in WORD_CATEGORIES:
        hits = [term for term, pattern in spec["rules"] if pattern.search(message)]

        if spec["name"] == "Money and prize bait" and CURRENCY_PATTERN.search(message):
            hits.append("currency amount")

        categories.append({
            "name": spec["name"],
            "cap": spec["cap"],
            "points": min(spec["cap"], len(hits) * spec["each"]),
            "hits": hits,
        })

    flagged = [row for row in links["rows"] if row["level_key"] != "safe"]
    categories.append({
        "name": "Suspicious links",
        "cap": 20,
        "points": min(20, links["dangerous"] * 12 + links["caution"] * 6),
        "hits": [row["host"] + " (" + row["level"].lower() + ")" for row in flagged],
    })

    style_hits = []
    style_points = 0

    if EXCLAMATION_PATTERN.search(message):
        style_hits.append("repeated exclamation marks")
        style_points += 3

    if len(CAPS_PATTERN.findall(message)) >= 3:
        style_hits.append("many ALL-CAPS words")
        style_points += 4

    if CONTACT_PATTERN.search(message):
        style_hits.append("asks you to call, text, or reply")
        style_points += 3

    categories.append({
        "name": "Pressure style",
        "cap": 10,
        "points": min(10, style_points),
        "hits": style_hits,
    })

    for category in categories:
        category["percent"] = round(category["points"] / category["cap"] * 100)

    score = min(100, sum(category["points"] for category in categories))
    level_key, level = threat_level(score)

    return {
        "score": score,
        "level": level,
        "level_key": level_key,
        "categories": categories,
        "radar_labels": [category["name"] for category in categories],
        "radar_values": [
            RADAR_FLOOR + round(category["percent"] * (100 - RADAR_FLOOR) / 100)
            for category in categories
        ],
    }


# ---------------------------------------------------------------------------
# Explainable AI: which words pushed the SVM toward spam or ham
# ---------------------------------------------------------------------------

def collect_linear_parts(estimator, found):
    """Walk the model and collect (vectorizer, coefficients, classes) tuples."""
    if hasattr(estimator, "calibrated_classifiers_"):
        for calibrated in estimator.calibrated_classifiers_:
            inner = getattr(calibrated, "estimator", None) or getattr(calibrated, "base_estimator", None)
            if inner is not None:
                collect_linear_parts(inner, found)
        return

    steps = getattr(estimator, "steps", None)
    if steps:
        vectorizer = None
        for _, step in steps:
            if hasattr(step, "get_feature_names_out") and hasattr(step, "transform"):
                vectorizer = step
                break

        final = steps[-1][1]
        if vectorizer is not None and hasattr(final, "coef_"):
            found.append((vectorizer, np.asarray(final.coef_).reshape(-1), list(final.classes_)))


def build_explain_parts():
    parts = []

    try:
        found = []
        collect_linear_parts(model, found)

        for vectorizer, coefficients, classes in found:
            labels = [str(label).lower() for label in classes]
            if len(labels) != 2 or "spam" not in labels:
                continue

            sign = 1.0 if labels.index("spam") == 1 else -1.0
            parts.append((vectorizer, coefficients * sign, vectorizer.get_feature_names_out()))
    except Exception:
        app.logger.exception("Could not read model weights for the explanation")

    return parts


EXPLAIN_PARTS = build_explain_parts()


def explain_prediction(message, top_n=6):
    if not EXPLAIN_PARTS:
        return None

    totals = {}

    try:
        for vectorizer, coefficients, names in EXPLAIN_PARTS:
            row = vectorizer.transform([message])
            for column, value in zip(row.indices, row.data):
                word = str(names[column])
                totals[word] = totals.get(word, 0.0) + float(value) * float(coefficients[column])
    except Exception:
        app.logger.exception("Explanation failed")
        return None

    count = len(EXPLAIN_PARTS)
    contributions = {word: value / count for word, value in totals.items() if abs(value) > 1e-9}

    if not contributions:
        return None

    spam_items = sorted(((w, v) for w, v in contributions.items() if v > 0), key=lambda i: -i[1])
    ham_items = sorted(((w, v) for w, v in contributions.items() if v < 0), key=lambda i: i[1])

    spam_total = sum(v for _, v in spam_items)
    ham_total = sum(-v for _, v in ham_items)
    biggest = max([v for _, v in spam_items] + [-v for _, v in ham_items] + [1e-9])

    def build(items):
        return [
            {
                "word": word,
                "weight": abs(value),
                "percent": round(min(100.0, abs(value) / biggest * 100), 1),
            }
            for word, value in items[:top_n]
        ]

    return {
        "spam": build(spam_items),
        "ham": build(ham_items),
        "spam_share": round(spam_total / (spam_total + ham_total) * 100) if (spam_total + ham_total) else 0,
    }


# ---------------------------------------------------------------------------
# Email (.eml) parsing and sender checks
# ---------------------------------------------------------------------------

def strip_html(markup):
    markup = re.sub(r"(?is)<(script|style).*?</\1>", " ", markup)
    markup = re.sub(r"(?s)<[^>]+>", " ", markup)
    return re.sub(r"\s+", " ", unescape(markup)).strip()


def parse_eml(raw_bytes):
    message = BytesParser(policy=policy.default).parsebytes(raw_bytes)

    subject = str(message.get("subject", "") or "").strip()
    body_part = message.get_body(preferencelist=("plain", "html"))
    body = ""
    html_links = []

    if body_part is not None:
        body = body_part.get_content()
        if body_part.get_content_type() == "text/html":
            html_links = re.findall(r"""href=["'](https?://[^"']+)["']""", body, re.IGNORECASE)
            body = strip_html(body)

    text = (subject + "\n\n" + body).strip()
    if html_links:
        text += "\n\nLinks in this email:\n" + "\n".join(dict.fromkeys(html_links))

    return message, subject, text


def address_domain(address):
    address = (address or "").strip().strip("<>")
    return address.rsplit("@", 1)[-1].lower() if "@" in address else ""


def same_organisation(first, second):
    return bool(first and second) and (
        first == second or first.endswith("." + second) or second.endswith("." + first)
    )


def build_header_report(message, subject):
    from_name, from_addr = parseaddr(str(message.get("from", "") or ""))
    _, reply_addr = parseaddr(str(message.get("reply-to", "") or ""))
    return_path = str(message.get("return-path", "") or "").strip().strip("<>")

    from_domain = address_domain(from_addr)
    reply_domain = address_domain(reply_addr)
    return_domain = address_domain(return_path)

    findings = []

    def add(level, text):
        findings.append({"level": level, "text": text})

    if not from_addr:
        add("bad", "The email has no readable From address.")
    else:
        flags, points = inspect_domain(from_domain)
        if points >= 3:
            add("bad", "The sender domain " + from_domain + " looks suspicious: " + "; ".join(flags).lower() + ".")
        else:
            add("ok", "The sender domain " + from_domain + " has no obvious warning signs.")

    for brand, official in BRANDS.items():
        if brand in (from_name or "").lower() and from_domain and \
                not any(domain_matches(from_domain, domain) for domain in official):
            add("bad", "The display name mentions " + brand.capitalize() +
                " but the address comes from " + from_domain + ".")
            break

    if reply_domain:
        if same_organisation(from_domain, reply_domain):
            add("ok", "Replies go to the same organisation as the sender.")
        else:
            add("bad", "Replies are redirected to a different domain (" + reply_domain + ").")

    if return_domain and from_domain and not same_organisation(from_domain, return_domain):
        add("warn", "The Return-Path domain (" + return_domain +
            ") differs from the From domain, which is common in spoofed mail.")

    auth_text = " ".join(
        str(value) for header in ("authentication-results", "received-spf", "arc-authentication-results")
        for value in message.get_all(header, [])
    ).lower()

    results = {}
    for name in ("spf", "dkim", "dmarc"):
        match = re.search(r"\b" + name + r"=(\w+)", auth_text)
        if not match and name == "spf":
            match = re.search(r"^\s*(pass|fail|softfail|neutral|none)\b", auth_text)
        if match:
            results[name] = match.group(1)

    if results:
        for name, value in results.items():
            if value == "pass":
                add("ok", name.upper() + " check passed.")
            elif value in ("fail", "permerror", "temperror"):
                add("bad", name.upper() + " check failed (" + value + ").")
            else:
                add("warn", name.upper() + " check was inconclusive (" + value + ").")
    else:
        add("warn", "No SPF, DKIM, or DMARC results were found, so the sender cannot be verified from the headers.")

    if any(item["level"] == "bad" for item in findings):
        verdict_key, verdict = "danger", "Suspicious sender"
    elif any(item["level"] == "warn" for item in findings):
        verdict_key, verdict = "caution", "Needs caution"
    else:
        verdict_key, verdict = "safe", "Looks legitimate"

    return {
        "from_name": from_name or "Not provided",
        "from_addr": from_addr or "Not provided",
        "reply_to": reply_addr or "Not provided",
        "return_path": return_path or "Not provided",
        "subject": subject or "No subject",
        "findings": findings,
        "verdict": verdict,
        "verdict_key": verdict_key,
    }


# ---------------------------------------------------------------------------
# Full analysis of one message
# ---------------------------------------------------------------------------

def analyze_message(message):
    ((is_spam, confidence),) = classify_many([message])

    highlighted_parts, frequency = analyze_spam_words(message)
    links = inspect_links(message)
    threat = analyze_threat(message, links)
    explanation = explain_prediction(message)

    return {
        "message": message,
        "is_spam": is_spam,
        "prediction": "Spam" if is_spam else "Ham",
        "confidence": confidence,
        "highlighted_parts": highlighted_parts,
        "frequency": frequency,
        "links": links,
        "threat": threat,
        "explanation": explanation,
    }


def build_advice(result, header_report):
    tips = []
    threat = result["threat"]
    links = result["links"]
    level = threat["level_key"]

    if header_report and header_report["verdict_key"] == "danger":
        tips.append({"level": "bad", "icon": "🎭", "title": "The sender looks spoofed",
                     "text": "The email headers show warning signs. Do not trust the sender name, even if it looks familiar."})

    if result["is_spam"]:
        tips.append({"level": "bad", "icon": "🚫", "title": "Do not reply, click, or call",
                     "text": "The model classified this as spam. Replying confirms your address is active and invites more scams."})
    elif level in ("high", "critical"):
        tips.append({"level": "warn", "icon": "⚠️", "title": "Classified as ham, but the content looks risky",
                     "text": "The model and the rule-based radar disagree. Treat this message with extra care and verify it another way."})
    elif level == "medium":
        tips.append({"level": "warn", "icon": "🔍", "title": "Mostly fine, with a few warning signs",
                     "text": "Check the sender and any links before acting on this message."})
    else:
        tips.append({"level": "good", "icon": "✅", "title": "No strong warning signs",
                     "text": "The model says ham and the threat radar found little. No tool is perfect, so stay careful with unexpected requests."})

    if links["dangerous"]:
        tips.append({"level": "bad", "icon": "🔗", "title": "Do not open the dangerous links",
                     "text": "Some links imitate trusted brands, hide their destination, or use risky domains. Go to the real website by typing its address yourself."})
    elif links["caution"]:
        tips.append({"level": "warn", "icon": "🔗", "title": "Check links before opening them",
                     "text": "Some links have minor warning signs such as shorteners or unencrypted HTTP."})

    phishing = next((c for c in threat["categories"] if c["name"] == "Phishing terms"), None)
    if phishing and phishing["hits"]:
        tips.append({"level": "warn", "icon": "🔐", "title": "Never share OTPs, PINs, or passwords",
                     "text": "Real banks and services never ask for these by message or email."})

    if result["is_spam"] or level in ("high", "critical"):
        tips.append({"level": "info", "icon": "📣", "title": "Report and delete it",
                     "text": "Mark it as spam or phishing in your mail app. In India you can also forward SMS scams to 1909 or report them at cybercrime.gov.in."})

    return tips


def build_speech(result):
    threat = result["threat"]
    sentence = (
        "This message is classified as " + result["prediction"].lower() +
        " with " + str(round(result["confidence"])) + " percent confidence. "
        "The threat level is " + threat["level"].lower() +
        ", with a score of " + str(threat["score"]) + " out of 100. "
    )

    words = sum(count for _, count in result["frequency"])
    sentence += (str(words) + " suspicious words were found. ") if words else "No suspicious words were found. "

    links = result["links"]
    if links["total"]:
        sentence += str(links["total"]) + " links were checked, " + str(links["dangerous"]) + " dangerous. "

    return sentence.strip()


def build_report_data(result):
    explanation = result["explanation"]

    return {
        "prediction": result["prediction"],
        "confidence": round(result["confidence"], 2),
        "threat_score": result["threat"]["score"],
        "threat_level": result["threat"]["level"],
        "categories": [
            {"name": c["name"], "points": c["points"], "cap": c["cap"], "hits": c["hits"]}
            for c in result["threat"]["categories"]
        ],
        "words": [{"word": word, "count": count} for word, count in result["frequency"]],
        "links": [
            {"url": row["url"], "level": row["level"], "flags": row["flags"]}
            for row in result["links"]["rows"]
        ],
        "signals": {
            "spam": [item["word"] for item in explanation["spam"]] if explanation else [],
            "ham": [item["word"] for item in explanation["ham"]] if explanation else [],
        },
        "message": result["message"][:3000],
    }


# ---------------------------------------------------------------------------
# Scan history (SQLite)
# ---------------------------------------------------------------------------

db_lock = threading.Lock()


def db_connect():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with db_lock, db_connect() as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created TEXT NOT NULL,
                message TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                threat_score INTEGER NOT NULL,
                threat_level TEXT NOT NULL,
                level_key TEXT NOT NULL,
                source TEXT NOT NULL
            )"""
        )


def record_scan(result, source):
    try:
        with db_lock, db_connect() as connection:
            connection.execute(
                """INSERT INTO scans
                   (created, message, label, confidence, threat_score, threat_level, level_key, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    result["message"],
                    result["prediction"],
                    round(result["confidence"], 2),
                    result["threat"]["score"],
                    result["threat"]["level"],
                    result["threat"]["level_key"],
                    source,
                ),
            )
            connection.execute(
                "DELETE FROM scans WHERE id <= (SELECT MAX(id) FROM scans) - ?",
                (MAX_STORED_SCANS,),
            )
    except sqlite3.Error:
        app.logger.exception("Could not save the scan to history")


def short_preview(text, limit=140):
    flat = re.sub(r"\s+", " ", text).strip()
    return flat if len(flat) <= limit else flat[:limit - 1] + "…"


def get_history():
    try:
        with db_connect() as connection:
            rows = connection.execute(
                "SELECT * FROM scans ORDER BY id DESC LIMIT ?", (HISTORY_LIMIT,)
            ).fetchall()
    except sqlite3.Error:
        return []

    history = []
    for row in rows:
        stamp = datetime.fromisoformat(row["created"])
        history.append({
            "time": stamp.strftime("%d %b, %H:%M"),
            "message": row["message"],
            "preview": short_preview(row["message"]),
            "label": row["label"],
            "confidence": f"{row['confidence']:.1f}",
            "threat": f"{row['threat_level']} · {row['threat_score']}",
            "level_key": row["level_key"],
            "source": row["source"],
        })
    return history


def get_stats():
    stats = {"total": 0, "spam": 0, "ham": 0}

    try:
        with db_connect() as connection:
            for row in connection.execute("SELECT label, COUNT(*) AS n FROM scans GROUP BY label"):
                key = "spam" if row["label"] == "Spam" else "ham"
                stats[key] += row["n"]
    except sqlite3.Error:
        pass

    stats["total"] = stats["spam"] + stats["ham"]
    return stats


def csv_safe(value):
    """Stop spreadsheet programs from running text that starts like a formula."""
    text = str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


# ---------------------------------------------------------------------------
# Feedback (feedback.csv, used by retrain.py)
# ---------------------------------------------------------------------------

feedback_lock = threading.Lock()
FEEDBACK_COLUMNS = ["timestamp", "message", "predicted", "correct", "true_label"]


def save_feedback(message, predicted, correct):
    opposite = "Ham" if predicted == "Spam" else "Spam"
    true_label = predicted if correct else opposite

    with feedback_lock:
        is_new = not FEEDBACK_PATH.exists() or FEEDBACK_PATH.stat().st_size == 0

        with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if is_new:
                writer.writerow(FEEDBACK_COLUMNS)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                message, predicted, int(correct), true_label,
            ])


def get_feedback_stats():
    stats = {"total": 0, "wrong": 0}

    if not FEEDBACK_PATH.exists():
        return stats

    try:
        with feedback_lock, open(FEEDBACK_PATH, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                stats["total"] += 1
                if str(row.get("correct", "1")) == "0":
                    stats["wrong"] += 1
    except (OSError, csv.Error):
        pass

    return stats


# ---------------------------------------------------------------------------
# Dataset, dashboard metrics and algorithm comparison
# ---------------------------------------------------------------------------

def find_column(columns, candidates):
    lookup = {str(column).strip().lower(): column for column in columns}
    for name in candidates:
        if name in lookup:
            return lookup[name]
    return None


def load_dataset():
    frame = pd.read_csv(DATASET_PATH, encoding="latin-1")

    label_col = find_column(frame.columns, ["v1", "label", "category", "class"])
    text_col = find_column(frame.columns, ["v2", "message", "text", "body"])

    if label_col is None or text_col is None:
        raise ValueError("spam.csv needs a label column (v1) and a message column (v2).")

    frame = frame[[label_col, text_col]].copy()
    frame.columns = ["label", "message"]
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame = frame.dropna()
    frame = frame[frame["label"].isin(["ham", "spam"])]
    # The SMS dataset has hundreds of repeated messages. Keeping them would put the
    # same text in both the training and test sets and inflate every score.
    return frame.drop_duplicates(subset="message")


_cache = {}
_cache_lock = threading.RLock()


def get_split():
    from sklearn.model_selection import train_test_split

    with _cache_lock:
        if "split" not in _cache:
            frame = load_dataset()
            _cache["split"] = train_test_split(
                frame["message"], frame["label"],
                test_size=0.2, random_state=42, stratify=frame["label"],
            )
        return _cache["split"]


def get_metrics():
    """Held-out evaluation on the 20% test split of spam.csv (cached).

    The saved .joblib was most likely trained on the whole dataset, so scoring it on
    part of that same data would look better than it really is. Instead, an identical
    copy of the model (same pipeline and settings) is trained on the 80% split only and
    scored on the unseen 20%. That is an honest estimate of how this model design performs.
    """
    with _cache_lock:
        if "metrics" in _cache:
            return _cache["metrics"]

        value = None

        try:
            from sklearn.base import clone
            from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

            X_train, X_test, y_train, y_test = get_split()
            held_out_model = clone(model).fit(X_train, y_train)
            actual = (y_test == "spam").astype(int).to_numpy()
            predicted = np.array([int(is_spam_label(label)) for label in held_out_model.predict(X_test)])

            tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
            precision, recall, f1, _ = precision_recall_fscore_support(
                actual, predicted, average="binary", zero_division=0
            )

            value = {
                "accuracy": float((tn + tp) / len(actual) * 100),
                "precision": float(precision * 100),
                "recall": float(recall * 100),
                "f1": float(f1 * 100),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "test_size": int(len(actual)),
            }
        except Exception:
            app.logger.exception("Could not compute model metrics")

        _cache["metrics"] = value
        return value


def run_comparison():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                                 roc_auc_score, roc_curve)
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC

    X_train, X_test, y_train, y_test = get_split()
    y_train_bin = (y_train == "spam").astype(int).to_numpy()
    y_test_bin = (y_test == "spam").astype(int).to_numpy()

    vectorizer = TfidfVectorizer()
    train_matrix = vectorizer.fit_transform(X_train)
    test_matrix = vectorizer.transform(X_test)

    specs = [
        ("Naive Bayes", MultinomialNB(), "proba"),
        ("Logistic Regression", LogisticRegression(max_iter=1000, C=10), "proba"),
        ("Support Vector Machine", LinearSVC(), "decision"),
    ]

    results = []

    for name, estimator, score_type in specs:
        started = time.perf_counter()
        estimator.fit(train_matrix, y_train_bin)
        train_ms = (time.perf_counter() - started) * 1000

        predicted = estimator.predict(test_matrix)
        scores = (estimator.predict_proba(test_matrix)[:, 1] if score_type == "proba"
                  else estimator.decision_function(test_matrix))

        tn, fp, fn, tp = confusion_matrix(y_test_bin, predicted, labels=[0, 1]).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_bin, predicted, average="binary", zero_division=0
        )

        fpr, tpr, _ = roc_curve(y_test_bin, scores)
        keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 120)).astype(int))

        results.append({
            "name": name,
            "accuracy": float((tn + tp) / len(y_test_bin) * 100),
            "precision": float(precision * 100),
            "recall": float(recall * 100),
            "f1": float(f1 * 100),
            "auc": float(roc_auc_score(y_test_bin, scores)),
            "fp": int(fp),
            "fn": int(fn),
            "train_ms": float(train_ms),
            "roc": [{"x": round(float(fpr[i]), 4), "y": round(float(tpr[i]), 4)} for i in keep],
        })

    best = max(results, key=lambda item: (item["f1"], -item["fp"]))

    return {"models": results, "best": best["name"], "test_size": int(len(y_test_bin))}


# ---------------------------------------------------------------------------
# Page context
# ---------------------------------------------------------------------------

def base_context(active_tab="analyze", **extra):
    context = {
        "active_tab": active_tab,
        "message": "",
        "stats": get_stats(),
        "feedback": get_feedback_stats(),
        "history": get_history(),
        "metrics": get_metrics(),
    }
    context.update(extra)
    return context


@app.route("/")
def home():
    return render_template("index.html", **base_context())


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("index.html", **base_context())

    message = request.form.get("message", "").strip()
    upload = request.files.get("eml_file")
    header_report = None
    source = "Text"

    if upload and upload.filename:
        try:
            parsed, subject, message = parse_eml(upload.read())
            header_report = build_header_report(parsed, subject)
            source = "Email file"
        except Exception:
            app.logger.exception("Could not read the .eml file")
            return render_template(
                "index.html",
                **base_context(error="That file could not be read as an email (.eml). Please try another file.")
            )

    message = message[:MAX_MESSAGE_LENGTH].strip()

    if not message:
        return render_template(
            "index.html",
            **base_context(error="Please enter a message to analyze.")
        )

    try:
        result = analyze_message(message)
        record_scan(result, source)

        return render_template(
            "index.html",
            **base_context(
                message=message,
                prediction=result["prediction"],
                confidence=result["confidence"],
                confidence_percentage=result["confidence"],
                highlighted_parts=result["highlighted_parts"],
                chart_labels=[word for word, _ in result["frequency"]],
                chart_counts=[count for _, count in result["frequency"]],
                suspicious_word_count=sum(count for _, count in result["frequency"]),
                speech_text=build_speech(result),
                report_data=build_report_data(result),
                advice=build_advice(result, header_report),
                threat=result["threat"],
                explanation=result["explanation"],
                links=result["links"],
                header_report=header_report,
            )
        )

    except Exception:
        app.logger.exception("Message analysis failed")
        return render_template(
            "index.html",
            **base_context(
                message=message,
                error="Unable to analyze this message. Please check your model and try again.",
            )
        )


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------

batch_store = OrderedDict()
batch_lock = threading.Lock()


def read_uploaded_csv(raw):
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise ValueError("The file encoding could not be read.")


def label_to_bool(value):
    text = str(value).strip().lower()
    if text in ("spam", "1", "true"):
        return True
    if text in ("ham", "0", "false", "not spam"):
        return False
    return None


@app.route("/batch", methods=["POST"])
def batch():
    upload = request.files.get("csv_file")

    def fail(text):
        return render_template("index.html", **base_context("batch", batch_error=text))

    if not upload or not upload.filename:
        return fail("Please choose a CSV file first.")

    try:
        frame = read_uploaded_csv(upload.read())
    except Exception:
        app.logger.exception("Could not read the CSV")
        return fail("That file could not be read as a CSV. Please check the format and try again.")

    text_col = find_column(frame.columns, ["message", "text", "body", "v2", "sms", "email", "content"])
    label_col = find_column(frame.columns, ["label", "v1", "category", "class", "target", "type"])

    if text_col is None:
        return fail("No message column found. Name a column message, text, body, or v2. "
                    "Columns found: " + ", ".join(str(c) for c in frame.columns[:8]))

    frame = frame.dropna(subset=[text_col]).copy()
    frame[text_col] = frame[text_col].astype(str).str.strip()
    frame = frame[frame[text_col] != ""]

    if frame.empty:
        return fail("The file has no messages to analyze.")

    truncated = len(frame) > MAX_BATCH_ROWS
    frame = frame.head(MAX_BATCH_ROWS)
    texts = [text[:MAX_MESSAGE_LENGTH] for text in frame[text_col]]

    try:
        predictions = classify_many(texts)
    except Exception:
        app.logger.exception("Batch classification failed")
        return fail("The messages could not be classified. Please try a different file.")

    rows = []
    for index, (text, (is_spam, confidence)) in enumerate(zip(texts, predictions), start=1):
        threat = analyze_threat(text, inspect_links(text))
        rows.append({
            "n": index,
            "message": text,
            "preview": short_preview(text),
            "is_spam": is_spam,
            "label": "Spam" if is_spam else "Ham",
            "confidence": round(confidence, 1),
            "threat": f"{threat['level']} · {threat['score']}",
            "threat_score": threat["score"],
            "threat_level": threat["level"],
            "level_key": threat["level_key"],
        })

    accuracy = None
    actual_labels = None

    if label_col is not None:
        actual_labels = [label_to_bool(value) for value in frame[label_col]]
        if all(value is not None for value in actual_labels):
            matches = sum(1 for row, actual in zip(rows, actual_labels) if row["is_spam"] == actual)
            accuracy = round(matches / len(rows) * 100, 2)

    output = io.StringIO()
    writer = csv.writer(output)
    header = ["message", "prediction", "confidence_percent", "threat_score", "threat_level"]
    if accuracy is not None:
        header.append("your_label")
    writer.writerow(header)

    for position, row in enumerate(rows):
        line = [csv_safe(row["message"]), row["label"], row["confidence"],
                row["threat_score"], row["threat_level"]]
        if accuracy is not None:
            line.append("Spam" if actual_labels[position] else "Ham")
        writer.writerow(line)

    token = uuid.uuid4().hex
    with batch_lock:
        batch_store[token] = output.getvalue()
        while len(batch_store) > MAX_STORED_BATCHES:
            batch_store.popitem(last=False)

    spam_count = sum(1 for row in rows if row["is_spam"])

    summary = {
        "filename": upload.filename,
        "total": len(rows),
        "spam": spam_count,
        "ham": len(rows) - spam_count,
        "high_threat": sum(1 for row in rows if row["level_key"] in ("high", "critical")),
        "spam_rate": round(spam_count / len(rows) * 100, 1),
        "accuracy": accuracy,
        "truncated": truncated,
        "shown": min(BATCH_PREVIEW_ROWS, len(rows)),
        "token": token,
        "rows": rows[:BATCH_PREVIEW_ROWS],
    }

    return render_template("index.html", **base_context("batch", batch=summary))


@app.route("/batch/download/<token>")
def batch_download(token):
    with batch_lock:
        content = batch_store.get(token) if re.fullmatch(r"[0-9a-f]{32}", token) else None

    if content is None:
        return Response("This download has expired. Please analyze the file again.", status=404)

    return Response(
        content, mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=mailguard-results.csv"},
    )


# ---------------------------------------------------------------------------
# History export / clear
# ---------------------------------------------------------------------------

@app.route("/history/export")
def history_export():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["time", "message", "verdict", "confidence_percent",
                     "threat_score", "threat_level", "source"])

    try:
        with db_connect() as connection:
            for row in connection.execute("SELECT * FROM scans ORDER BY id DESC"):
                writer.writerow([row["created"], csv_safe(row["message"]), row["label"],
                                 row["confidence"], row["threat_score"],
                                 row["threat_level"], row["source"]])
    except sqlite3.Error:
        app.logger.exception("History export failed")

    return Response(
        output.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=mailguard-history.csv"},
    )


@app.route("/history/clear", methods=["POST"])
def history_clear():
    with db_lock, db_connect() as connection:
        connection.execute("DELETE FROM scans")
    return redirect("/#history")


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

def api_payload(result):
    explanation = result["explanation"]

    return {
        "prediction": result["prediction"],
        "is_spam": result["is_spam"],
        "confidence": round(result["confidence"], 2),
        "threat": {"score": result["threat"]["score"], "level": result["threat"]["level"]},
        "suspicious_words": [{"word": word, "count": count} for word, count in result["frequency"]],
        "links": [
            {"url": row["url"], "level": row["level"], "flags": row["flags"]}
            for row in result["links"]["rows"]
        ],
        "explanation": {
            "toward_spam": [{"word": i["word"], "weight": round(i["weight"], 4)}
                            for i in explanation["spam"]] if explanation else [],
            "toward_ham": [{"word": i["word"], "weight": round(i["weight"], 4)}
                           for i in explanation["ham"]] if explanation else [],
        },
    }


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": 'Send JSON such as {"message": "text"}.'}), 400

    if "messages" in data:
        messages = data["messages"]

        if not isinstance(messages, list) or not messages:
            return jsonify({"error": '"messages" must be a non-empty list of strings.'}), 400
        if len(messages) > API_BATCH_LIMIT:
            return jsonify({"error": f"At most {API_BATCH_LIMIT} messages per request."}), 400
        if not all(isinstance(item, str) and item.strip() for item in messages):
            return jsonify({"error": "Every message must be a non-empty string."}), 400

        results = []
        for item in messages:
            result = analyze_message(item.strip()[:MAX_MESSAGE_LENGTH])
            record_scan(result, "API")
            results.append(api_payload(result))

        return jsonify({"results": results})

    message = data.get("message")

    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": 'Provide a non-empty "message" string.'}), 400

    result = analyze_message(message.strip()[:MAX_MESSAGE_LENGTH])
    record_scan(result, "API")
    return jsonify(api_payload(result))


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "Send JSON with message, predicted, and correct."}), 400

    message = data.get("message")
    predicted = str(data.get("predicted", "")).strip().capitalize()
    correct = data.get("correct")

    if isinstance(correct, str):
        correct = correct.strip().lower() in ("1", "true", "yes")

    if not isinstance(message, str) or not message.strip():
        return jsonify({"ok": False, "error": "A message is required."}), 400
    if predicted not in ("Spam", "Ham"):
        return jsonify({"ok": False, "error": 'predicted must be "Spam" or "Ham".'}), 400
    if not isinstance(correct, bool):
        return jsonify({"ok": False, "error": "correct must be true or false."}), 400

    try:
        save_feedback(message.strip()[:MAX_MESSAGE_LENGTH], predicted, correct)
    except OSError:
        app.logger.exception("Could not write feedback.csv")
        return jsonify({"ok": False, "error": "Could not save feedback."}), 500

    return jsonify({"ok": True})


@app.route("/api/stats")
def api_stats():
    stats = get_stats()
    stats["spam_rate"] = round(stats["spam"] / stats["total"] * 100, 1) if stats["total"] else 0.0
    stats["feedback"] = get_feedback_stats()
    return jsonify(stats)


@app.route("/api/compare")
def api_compare():
    with _cache_lock:
        cached = _cache.get("comparison")

        if cached is None:
            try:
                cached = run_comparison()
            except Exception as error:
                app.logger.exception("Algorithm comparison failed")
                return jsonify({"error": str(error) or "Comparison failed."}), 500

            _cache["comparison"] = cached

    return jsonify(cached)


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "model_loaded": True, "explanations": bool(EXPLAIN_PARTS)})


@app.errorhandler(413)
def too_large(_error):
    text = "That file is too large. The limit is 8 MB."

    if request.path.startswith("/api/"):
        return jsonify({"error": text}), 413

    if request.path.startswith("/batch"):
        return render_template("index.html", **base_context("batch", batch_error=text)), 413

    return render_template("index.html", **base_context(error=text)), 413


init_db()

if __name__ == "__main__":
    app.run(debug=True)