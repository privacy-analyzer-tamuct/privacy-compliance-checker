from flask import Flask, render_template, request, jsonify
from rdflib import Graph, URIRef, RDF, RDFS, OWL, Literal
from pathlib import Path
import json
import re
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from groq import Groq

# ====== NEW: pull in the step modules ======
import step2_state_detector as state_detector
import step4a_timestamps as ts
import step4b_regulation_update as reg_update
# Step 1 is exposed via a small wrapper below; the heavy work still
# lives in step1_add_regulation.py.
import step1_add_regulation as reg_add
import step3_agentic_weighted_grader as wgrader

app = Flask(__name__)
app.register_blueprint(wgrader.bp)

# ---- PDF Upload Settings ----
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
MAX_UPLOAD_MB = 10
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# -------------------- Config --------------------
CFG = json.load(open("config.json", "r", encoding="utf-8"))
ONTO_PATH = Path(CFG["ontology_path"]).resolve()
MANUFACTURER_CLS = URIRef(CFG["manufacturer_class_iri"])
POLICY_PROP = URIRef(CFG["policy_property_iri"])

TOP_K = int(CFG.get("top_k", 8))
LAW_PREDICATES = [URIRef(p) for p in CFG.get("law_annotation_predicates", [])]
LAWS = CFG.get("laws", [])
COVERAGE_THRESHOLD = float(CFG.get("coverage_threshold", 0.15))
MISSING_CLASS_SIM_THRESHOLD = COVERAGE_THRESHOLD
SIMILARITY_METHOD = CFG.get("similarity_method", "tfidf").lower()
EMBEDDING_MODEL_NAME = CFG.get("embedding_model_name", "all-MiniLM-L6-v2")
USER_ADDED_PROP = URIRef("http://example.org/onto.owl#userAddedManufacturer")

# NEW: predicate to stamp which states a manufacturer applies to
APPLIES_TO_STATE_PROP = URIRef("http://example.org/onto.owl#appliesToState")


# -------------------- Name cleanup --------------------
ALLOWED_MANUFACTURERS = {
    "ADT", "ATT", "Brainly", "Bloomberg", "Dodge", "Emerson", "EOS", "Fitbit",
    "NYTimes", "Panasonic", "Puma", "Ring", "UPS", "Verizon", "Vivint",
}
SPECIAL_NAME_FIXES = {
    "att": "AT&T", "adt": "ADT", "ups": "UPS",
    "nytimes": "NYTimes", "eos": "EOS", "ebay": "eBay",
}


def clean_manufacturer_name(name: str) -> str:
    lower = name.lower().strip()
    if lower in SPECIAL_NAME_FIXES:
        return SPECIAL_NAME_FIXES[lower]
    return name.title()


def clean_class_label(label: str) -> str:
    fixes = {
        "network interface": "Network Interface",
        "iot device": "IoT Device",
        "io tdevice": "IoT Device",
        "iotdevice": "IoT Device",
    }
    return fixes.get(label.lower().strip(), label)


# -------------------- Load graph --------------------
g = Graph()
print(f"[load] {ONTO_PATH}")
g.parse(str(ONTO_PATH))


# -------------------- Helpers (unchanged) --------------------
def local_name(iri: str) -> str:
    s = iri
    if "#" in s:
        s = s.rsplit("#", 1)[1]
    s = s.rstrip("/").rsplit("/", 1)[-1]
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return s.replace("_", " ").replace("-", " ")


def extract_law_segments(value: str):
    segments_by_law = {law["id"]: [] for law in LAWS}
    if not value:
        return {}
    pieces = re.split(r'(?<=[.!?])\s+|[\n\r]+|;+', value)
    for piece in pieces:
        plow = piece.lower().strip()
        if not plow:
            continue
        for law in LAWS:
            lid = law["id"]
            for kw in law.get("keywords", []):
                if kw.lower() in plow:
                    segments_by_law[lid].append(piece.strip())
                    break
    return {lid: segs for lid, segs in segments_by_law.items() if segs}


# -------------------- Build class corpus (unchanged logic) --------------------
class_iris, class_labels, class_texts = [], [], []
class_descs, class_to_laws = {}, {}

for c in g.subjects(RDF.type, OWL.Class):
    if not isinstance(c, URIRef):
        continue
    c_iri = str(c)
    law_ids_for_class = set()
    law_snippets = []
    for pred in LAW_PREDICATES:
        for obj in g.objects(c, pred):
            text = str(obj)
            segments_by_law = extract_law_segments(text)
            for lid, segs in segments_by_law.items():
                law_ids_for_class.add(lid)
                law_snippets.extend(segs)
    if not law_ids_for_class:
        continue
    labels = [str(o) for o in g.objects(c, RDFS.label)]
    comments = [str(o) for o in g.objects(c, RDFS.comment)]
    if not labels:
        labels = [clean_class_label(local_name(c_iri))]
    label_text = clean_class_label(" / ".join(labels))
    combined = " ".join(labels + comments + law_snippets + [local_name(c_iri)])
    class_desc = " ".join(comments).strip() if comments else " ".join(law_snippets).strip()
    class_iris.append(c_iri)
    class_labels.append(label_text)
    class_texts.append(combined)
    class_descs[c_iri] = class_desc
    class_to_laws[c_iri] = law_ids_for_class

if not class_texts:
    raise SystemExit("No eligible classes found with law annotations matching configured keywords.")

# Deduplicate
seen = set()
dedup_iris, dedup_labels, dedup_texts = [], [], []
dedup_descs, dedup_laws = {}, {}
for i, c_iri in enumerate(class_iris):
    clean = class_labels[i].lower().strip()
    if clean in seen:
        continue
    seen.add(clean)
    dedup_iris.append(c_iri)
    dedup_labels.append(class_labels[i])
    dedup_texts.append(class_texts[i])
    dedup_descs[c_iri] = class_descs[c_iri]
    dedup_laws[c_iri] = class_to_laws[c_iri]
class_iris, class_labels, class_texts = dedup_iris, dedup_labels, dedup_texts
class_descs, class_to_laws = dedup_descs, dedup_laws

iri_to_idx = {c_iri: idx for idx, c_iri in enumerate(class_iris)}
law_to_class_idxs = {law["id"]: [] for law in LAWS}
for c_iri, law_ids in class_to_laws.items():
    idx = iri_to_idx.get(c_iri)
    if idx is None:
        continue
    for lid in law_ids:
        if lid in law_to_class_idxs:
            law_to_class_idxs[lid].append(idx)

for law in LAWS:
    lid = law["id"]
    print(f"[init] Law {lid} has {len(law_to_class_idxs.get(lid, []))} related classes")


# -------------------- Similarity model --------------------
if SIMILARITY_METHOD == "tfidf":
    print("[init] Using TF-IDF similarity")
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english",
                                  ngram_range=(1, 2), max_features=5000)
    X_classes = vectorizer.fit_transform(class_texts)
    class_embeddings = None
elif SIMILARITY_METHOD == "bert":
    print(f"[init] Using BERT embeddings ({EMBEDDING_MODEL_NAME})")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    class_embeddings = model.encode(class_texts, convert_to_numpy=True,
                                     normalize_embeddings=True)
    vectorizer = None
    X_classes = None
else:
    raise ValueError(f"Unknown similarity_method: {SIMILARITY_METHOD}")


# -------------------- Manufacturers --------------------
manufacturers = []
q = f"""
SELECT DISTINCT ?inst WHERE {{
  ?inst a ?t .
  ?t rdfs:subClassOf* <{CFG["manufacturer_class_iri"]}> .
}}
"""
for row in g.query(q, initNs={"rdfs": RDFS}):
    inst = row.inst
    iri = str(inst)
    labels = [str(o) for o in g.objects(inst, RDFS.label)]
    name = clean_manufacturer_name(labels[0]) if labels else clean_manufacturer_name(local_name(iri))
    policies = [str(o) for o in g.objects(inst, POLICY_PROP) if isinstance(o, Literal)]
    policy = max(policies, key=len) if policies else ""
    user_added = any(g.objects(inst, USER_ADDED_PROP))
    manufacturers.append({"iri": iri, "name": name, "policy": policy, "user_added": user_added})

manufacturers.sort(key=lambda m: m["name"].lower())


def is_allowed_name(name: str) -> bool:
    nlow = name.lower()
    return any(allowed.lower() in nlow for allowed in ALLOWED_MANUFACTURERS)


filtered = [m for m in manufacturers if m.get("user_added") or is_allowed_name(m["name"])]
if filtered:
    manufacturers = filtered
    print(f"[init] Loaded {len(manufacturers)} manufacturers after filtering.")
else:
    print("[init] WARNING: filter removed all manufacturers; using full list instead.")


def generate_manufacturer_iri(name: str) -> URIRef:
    base = str(MANUFACTURER_CLS).split("#")[0] + "#"
    slug = re.sub(r"\W+", "_", name.strip()) or "Manufacturer"
    candidate = URIRef(base + slug)
    i = 1
    while (candidate, None, None) in g:
        candidate = URIRef(base + f"{slug}_{i}")
        i += 1
    return candidate


def find_existing_manufacturer_by_name(name: str):
    """For re-upload detection: look up an existing manufacturer by name."""
    target = name.lower().strip()
    for m in manufacturers:
        if m["name"].lower().strip() == target:
            return m
    return None

# ---- SCORING !STATIC for now
""" agentigraph paper findings
#--- **idea 1** - weighted classes determined by agents that affect compliance scores 
                    - (ex strong preconfig authentication > transducer element )
                - how much an IoT manufacturer neglects one aspect of regulation that is more detrimental than another

#--- idea 2 - /for chat imp/ - MULTI-HOP reasoning: llm decides how many hops to conclude evaluation of anaylses
                - surgically pulls out kg content LLM needs to answer questions for efficeincy and breaks it down with agents
                - pertty ideal for complex queries/more dynamic
                    v- LLM LOOKS for missing classes
                    v- Result Returned
                    v- rescores class
                    ^v- Result returned or loops back if unsure
                - FOR MORE ACCURATE GAP IDENTIFICATION 
"""
# -------------------- Scoring (unchanged) --------------------
def rank_classes_for_policy(policy_text: str, state: str = "all"):
    if not policy_text or not policy_text.strip():
        return [], None
    if SIMILARITY_METHOD == "tfidf":
        q_vec = vectorizer.transform([policy_text])
        sims = cosine_similarity(q_vec, X_classes)[0]
    elif SIMILARITY_METHOD == "bert":
        q_emb = model.encode([policy_text], convert_to_numpy=True,
                             normalize_embeddings=True)[0]
        sims = np.dot(class_embeddings, q_emb)
    else:
        raise ValueError(f"Unknown similarity_method: {SIMILARITY_METHOD}")
    idxs = [i for i, s in enumerate(sims) if float(s) >= COVERAGE_THRESHOLD]
    idxs.sort(key=lambda i: sims[i], reverse=True)
    top_classes = [{
        "class_iri": class_iris[i],
        "class_label": class_labels[i],
        "class_desc": class_descs.get(class_iris[i], ""),
    } for i in idxs]
    return top_classes, sims


def compute_law_coverage(sims, state: str = "all"):
    if sims is None or not LAWS:
        return [], [law["id"] for law in LAWS], 0.0
    # Filter laws by state if requested
    if state and state != "all":
        applicable = state_detector.applicable_laws([state], LAWS)
        applicable_ids = {law["id"] for law in applicable}
        active_laws = [law for law in LAWS if law["id"] in applicable_ids]
    else:
        active_laws = LAWS
    law_coverage, missing_laws = [], []
    total_classes, total_above = 0, 0
    for law in active_laws:
        lid = law["id"]
        class_idxs = law_to_class_idxs.get(lid, [])
        if not class_idxs:
            law_coverage.append({"id": lid, "label": law["label"],
                                 "coverage_percent": 0.0,
                                 "num_classes": 0, "num_above_threshold": 0})
            missing_laws.append(lid)
            continue
        scores = [float(sims[i]) for i in class_idxs]
        above = [s for s in scores if s >= COVERAGE_THRESHOLD]
        coverage_percent = 100.0 * len(above) / len(class_idxs)
        law_coverage.append({"id": lid, "label": law["label"],
                             "coverage_percent": round(coverage_percent, 2),
                             "num_classes": len(class_idxs),
                             "num_above_threshold": len(above)})
        if len(above) == 0:
            missing_laws.append(lid)
        total_classes += len(class_idxs)
        total_above += len(above)
    overall_percent = round(100.0 * total_above / total_classes, 2) if total_classes > 0 else 0.0
    return law_coverage, missing_laws, overall_percent


def compute_missing_classes(sims, state: str = "all"):
    if sims is None:
        return []
    if state and state != "all":
        applicable = state_detector.applicable_laws([state], LAWS)
        applicable_ids = {law["id"] for law in applicable}
        active_laws = [law for law in LAWS if law["id"] in applicable_ids]
    else:
        active_laws = LAWS
    missing = []
    for law in active_laws:
        lid = law["id"]
        for idx in law_to_class_idxs.get(lid, []):
            if float(sims[idx]) < COVERAGE_THRESHOLD:
                c_iri = class_iris[idx]
                missing.append({
                    "class_iri": c_iri,
                    "class_label": class_labels[idx],
                    "class_desc": class_descs.get(c_iri, ""),
                    "law_id": lid, "law_label": law["label"],
                })
    return missing


# -------------------- SWRL / SPARQL helpers --------------------
# SWRL inferencing flag — set True if your ontology uses SWRL rules
# and you have a SWRL-capable reasoner active; False otherwise.
SWRL_ACTIVE = False

SWRL_COVERS_PRED = URIRef("http://example.org/onto.owl#swrl_covers")


def get_swrl_covered_classes(manufacturer_iri: str) -> set:
    """
    Return the set of class IRIs that SWRL rules have inferred the
    given manufacturer covers. If SWRL is not active, returns empty set.
    """
    if not SWRL_ACTIVE:
        return set()
    inst = URIRef(manufacturer_iri)
    return {str(obj) for obj in g.objects(inst, SWRL_COVERS_PRED)}


def get_hybrid_tier(bert_covered: bool, swrl_decision: bool) -> str:
    """
    Combine BERT similarity result with SWRL inference into a single tier label.
      BERT ✓  SWRL ✓  -> "confirmed"
      BERT ✓  SWRL ✗  -> "bert_only"
      BERT ✗  SWRL ✓  -> "swrl_only"   (shouldn't appear in top_classes)
      BERT ✗  SWRL ✗  -> "neither"
    """
    if bert_covered and swrl_decision:
        return "confirmed"
    if bert_covered:
        return "bert_only"
    if swrl_decision:
        return "swrl_only"
    return "neither"

# -------- VALIDATION !STATIC
# --- idea 1 - 
def sparql_validate_manufacturer(manufacturer_iri: str, sims) -> dict:
    """
    Cross-validate BERT similarity scores against the ontology structure
    using SPARQL. Returns an agreement summary used by /detail and /sparql_validate_all.
    """
    inst = URIRef(manufacturer_iri)
    # Query: which classes does the ontology directly link this manufacturer to?
    sparql_covered = set()
    try:
        q = f"""
        SELECT DISTINCT ?cls WHERE {{
            <{manufacturer_iri}> a ?cls .
            ?cls a <{OWL.Class}> .
        }}
        """
        for row in g.query(q):
            sparql_covered.add(str(row.cls))
    except Exception:
        pass

    bert_covered = set()
    if sims is not None:
        for i, score in enumerate(sims):
            if float(score) >= COVERAGE_THRESHOLD:
                bert_covered.add(class_iris[i])

    agreed = len(bert_covered & sparql_covered)
    total_pairs = len(bert_covered | sparql_covered)

    return {
        "agreed": agreed,
        "total_pairs": total_pairs,
        "agreement_rate": round(100.0 * agreed / total_pairs, 2) if total_pairs > 0 else 0.0,
        "bert_only": list(bert_covered - sparql_covered)[:5],
        "sparql_only": list(sparql_covered - bert_covered)[:5],
    }


# ==========================================================================
# ROUTES
# ==========================================================================
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/list")
def list_instances():
    return jsonify([{"iri": m["iri"], "name": m["name"]} for m in manufacturers])





# ---------------------------------------------------------------
# /add_manufacturer — now with timestamps + state detection
# ---------------------------------------------------------------
@app.post("/add_manufacturer")
def add_manufacturer():
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    policy = (data.get("policy") or "").strip()
    # NEW: optional list of state codes the user checked on the UI
    selected_states = data.get("selected_states") or []

    if not name or not policy:
        return jsonify({"error": "Both name and policy are required."}), 400

    # Detect whether this is a re-upload (same name as an existing manufacturer)
    existing = find_existing_manufacturer_by_name(name)

    if existing:
        inst_iri = URIRef(existing["iri"])
    else:
        inst_iri = generate_manufacturer_iri(name)
        g.add((inst_iri, RDF.type, MANUFACTURER_CLS))
        g.add((inst_iri, RDFS.label, Literal(name)))
        g.add((inst_iri, USER_ADDED_PROP, Literal(True)))

    # ---- STEP 4a: upsert policy with timestamps ----
    ts_info = ts.upsert_policy(g, inst_iri, POLICY_PROP, policy)

    # ---- STEP 2: figure out which states apply ----
    auto = state_detector.detect_states_from_text(policy)
    if selected_states:
        final_states = selected_states
    else:
        final_states = [s["id"] for s in auto["detected"]]

    # Clear previous state stamps and write the new ones
    for old in list(g.objects(inst_iri, APPLIES_TO_STATE_PROP)):
        g.remove((inst_iri, APPLIES_TO_STATE_PROP, old))
    for sid in final_states:
        g.add((inst_iri, APPLIES_TO_STATE_PROP, Literal(sid)))

    applicable = state_detector.applicable_laws(final_states, LAWS)

    # Update in-memory list
    new_entry = {
        "iri": str(inst_iri),
        "name": clean_manufacturer_name(name),
        "policy": policy,
        "user_added": True,
    }
    if existing:
        # Replace the existing entry in the list
        manufacturers[:] = [m for m in manufacturers if m["iri"] != str(inst_iri)]
    manufacturers.append(new_entry)
    manufacturers.sort(key=lambda m: m["name"].lower())

    try:
        g.serialize(destination=str(ONTO_PATH), format="xml")
    except Exception as e:
        print("[error] Failed to save ontology:", e)
        return jsonify({
            "error": "Manufacturer saved in memory, but failed to update ontology file."
        }), 500

    return jsonify({
        "iri": new_entry["iri"],
        "name": new_entry["name"],
        # NEW fields in the response:
        "action": ts_info["action"],  # "created" or "updated"
        "created_at": ts_info["created_at"],
        "modified_at": ts_info["modified_at"],
        "had_previous_version": ts_info["previous_policy"] is not None,
        "auto_detected_states": auto,
        "selected_states": final_states,
        "applicable_laws": applicable,
    }), 200 if existing else 201


# ---------------------------------------------------------------
# NEW: /detect_states — preview what Step 2 would do without saving
# ---------------------------------------------------------------
@app.post("/detect_states")
def detect_states():
    data = request.get_json(force=True) or {}
    policy = (data.get("policy") or "").strip()
    if not policy:
        return jsonify({"error": "policy text required"}), 400

    auto = state_detector.detect_states_from_text(policy)
    applicable = state_detector.applicable_laws(
        [s["id"] for s in auto["detected"]], LAWS
    )
    return jsonify({
        "auto_detection": auto,
        "applicable_laws": applicable,
    })


# ---------------------------------------------------------------
# NEW: /manufacturer_history — timestamps + previous policy versions
# ---------------------------------------------------------------
@app.get("/manufacturer_history")
def manufacturer_history():
    iri = request.args.get("iri")
    if not iri:
        return jsonify({"error": "missing iri"}), 400
    hist = ts.get_policy_history(g, URIRef(iri), POLICY_PROP)
    return jsonify(hist)


# ---------------------------------------------------------------
# NEW: /upload_regulation — Step 1 exposed as an endpoint
# ---------------------------------------------------------------
@app.post("/upload_regulation")
def upload_regulation():
    """
    Accept a regulation PDF + metadata and run the Step 1 pipeline
    against the live ontology graph.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file field found."}), 400
    f = request.files["file"]
    law_id = (request.form.get("law_id") or "").strip()
    law_label = (request.form.get("law_label") or "").strip()

    if not f or not f.filename:
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Only PDF files are allowed."}), 400
    if not law_id or not law_label:
        return jsonify({"error": "law_id and law_label are required."}), 400

    filename = secure_filename(f.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(path)

    try:
        text = reg_add.read_pdf_text(path)
        sm = SentenceTransformer("all-MiniLM-L6-v2")
        matched, new_candidates = reg_add.extract_candidate_classes(text, sm)

        base_iri = str(MANUFACTURER_CLS).split("#")[0] + "#"
        annotated, created = [], []

        for m in matched:
            existing = reg_add.find_existing_class(g, m["label"])
            if existing is not None:
                reg_add.annotate_existing_class(g, existing, law_label, m["example_sentences"])
                annotated.append(m["label"])

        for n in new_candidates:
            if reg_add.find_existing_class(g, n["label"]) is not None:
                continue
            reg_add.create_new_class(g, base_iri, n["label"], law_label, n["example_sentences"])
            created.append(n["label"])

        # Register the law in config.json if new
        reg_add.register_law_in_config(
            "config.json", law_id, law_label, [law_label, law_id.replace("_", " ")]
        )

        g.serialize(destination=str(ONTO_PATH), format="xml")

        return jsonify({
            "law_id": law_id,
            "law_label": law_label,
            "classes_annotated": annotated,
            "classes_created": created,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.remove(path)
        except Exception: pass


# ---------------------------------------------------------------
# NEW: /update_regulation — Step 4b
# ---------------------------------------------------------------
@app.post("/update_regulation")
def update_regulation_route():
    """
    JSON body:
      {
        "law_id": "OR_HB_2395",
        "law_label": "Oregon HB 2395",
        "new_version": "2026-amendment",
        "updated_classes": [
           {"label": "Data Breach",
            "snippet": "Updated text mentioning 30-day notification."}
        ]
      }
    """
    data = request.get_json(force=True) or {}
    required = ["law_id", "law_label", "new_version", "updated_classes"]
    if not all(k in data for k in required):
        return jsonify({"error": f"Required: {required}"}), 400

    base_iri = str(MANUFACTURER_CLS).split("#")[0] + "#"
    result = reg_update.update_regulation(
        g, base_iri,
        law_label=data["law_label"],
        law_id=data["law_id"],
        new_version=data["new_version"],
        updated_classes=data["updated_classes"],
    )
    try:
        g.serialize(destination=str(ONTO_PATH), format="xml")
    except Exception as e:
        return jsonify({"error": f"Graph updated in memory but not saved: {e}", **result}), 500
    return jsonify(result)


# ---------------------------------------------------------------
# /extract_pdf — extract text and run agentic analysis
# ---------------------------------------------------------------
@app.post("/extract_pdf")
def extract_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file field found."}), 400
    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Only PDF files are allowed."}), 400
    filename = secure_filename(f.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(path)
    try:
        reader = PdfReader(path)
        texts = [page.extract_text() or "" for page in reader.pages]
        extracted = "\n\n".join(t for t in texts if t.strip()).strip()
    except Exception as e:
        return jsonify({"error": f"Failed to read PDF: {e}"}), 500
    finally:
        try: os.remove(path)
        except Exception: pass
    if not extracted:
        return jsonify({"error": "No text could be extracted from that PDF."}), 200
    # Agentic analysis: classify and segment the document
    analysis = analyze_policy_document(extracted)
    return jsonify({"text": extracted, "analysis": analysis}), 200
@app.get("/detail")
def detail():
    iri = request.args.get("iri")
    state = request.args.get("state", "all")

    if not iri:
        return jsonify({"error": "missing iri"}), 400

    match = next((m for m in manufacturers if m["iri"] == iri), None)
    if not match:
        inst = URIRef(iri)
        labels = [str(o) for o in g.objects(inst, RDFS.label)]
        name = clean_manufacturer_name(labels[0]) if labels else clean_manufacturer_name(local_name(iri))
        policies = [str(o) for o in g.objects(inst, POLICY_PROP) if isinstance(o, Literal)]
        policy = max(policies, key=len) if policies else ""
        match = {"iri": iri, "name": name, "policy": policy}

    top_classes, sims = rank_classes_for_policy(match["policy"], state)
    law_coverage, missing_laws, overall_percent = compute_law_coverage(sims, state)
    missing_classes = compute_missing_classes(sims, state)

    # SWRL hybrid tiers for top classes
    swrl_covered = get_swrl_covered_classes(iri)
    for cls in top_classes:
        bert_covered = True  # already above threshold to be in top_classes
        swrl_decision = cls["class_iri"] in swrl_covered
        cls["hybrid"] = get_hybrid_tier(bert_covered, swrl_decision)

    # SPARQL auto-validation — runs on every detail call
    sparql_validation = sparql_validate_manufacturer(iri, sims)

    return jsonify({
        "iri": match["iri"],
        "name": match["name"],
        "policy": match["policy"],
        "suggested_classes": top_classes,
        "law_coverage": law_coverage,
        "missing_laws": missing_laws,
        "missing_classes": missing_classes,
        "overall_coverage": overall_percent,
        "sparql_validation": sparql_validation,
        "swrl_active": SWRL_ACTIVE,
    })

def analyze_policy_document(text: str) -> dict:

    prompt = f"""
Classify the document AND extract policy segments.

Return JSON:
{{
  "document_type": "...",
  "manufacturer_name": "...",
  "confidence": "...",
  "segments": [
    {{
      "category": "...",
      "excerpt": "...",
      "confidence": "..."
    }}
  ]
}}

Document (first 3000 chars):
{text[:3000]}
"""

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800,
        )

        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        return json.loads(raw)

    except Exception as e:
        return {"error": str(e)}

@app.get("/sparql_validate_all")
def sparql_validate_all():
    """
    Run SPARQL vs BERT cross-validation for ALL manufacturers.
    Use this to generate your paper's agreement rate table.
    """
    results = []
    for m in manufacturers:
        _, sims = rank_classes_for_policy(m["policy"], "all")
        if sims is None:
            continue
        result = sparql_validate_manufacturer(m["iri"], sims)
        result["name"] = m["name"]
        results.append(result)

    # Compute overall agreement rate across all manufacturers
    total_pairs = sum(r["total_pairs"] for r in results)
    total_agreed = sum(r["agreed"] for r in results)
    overall_rate = round(
        100.0 * total_agreed / total_pairs, 2
    ) if total_pairs > 0 else 0.0

    return jsonify({
        "overall_agreement_rate": overall_rate,
        "total_pairs_evaluated": total_pairs,
        "total_agreed": total_agreed,
        "manufacturers": results
    }), 200

# ------------------------------------------------------------------aliases--------------------------------------------------------------------------
def expand_question_with_aliases(question: str) -> str:
    """Expand user questions with domain-specific aliases."""
    aliases = {
        # Password & Authentication
        "password": "password authentication credential login unique default",
        "login": "login authentication credential password access",
        "credential": "credential password authentication login",
        "auth": "authentication credential password login verification",
        
        # Updates & Patching
        "update": "update patch upgrade firmware software maintenance",
        "patch": "patch update upgrade fix software",
        "upgrade": "upgrade update patch firmware software",
        
        # Encryption & Security
        "encryption": "encryption encrypted secure crypto cryptographic protect",
        "encrypt": "encryption encrypted secure crypto cryptographic",
        "secure": "secure security encryption encrypted protection",
        "security": "security cybersecurity protection secure safe mechanism",
        
        # Data & Privacy
        "data protection": "data protection privacy security information safeguard compliance",
        "data privacy": "data privacy protection information personal confidential",
        "personal data": "personal data privacy information user sensitive",
        "data collection": "data collection gathering information privacy personal",
        "data": "data information privacy collection personal user",
        "privacy": "privacy data information personal protection collection",
        "personal protection": "personal data privacy information user",
        "collection": "collection data information gathering privacy",
        "protection": "protection security safeguard privacy data secure",
        
        # Network & Communication
        "network": "network communication interface connectivity protocol",
        "communication": "communication network interface connectivity protocol",
        "interface": "interface network communication connectivity",
        
        # Device & IoT
        "device": "device iot connected smart thing sensor transducer",
        "iot": "iot device connected smart internet things",
        "sensor": "sensor device transducer iot measurement",
        "smart": "smart iot device connected intelligent",
        
        # Access & Control
        "access control": "access control permission authorization authentication security",
        "access": "access control permission authorization authentication",
        "control": "control access management permission authorization",
    }
    
    expanded = question.lower()
    
    # Sort by length (longest first) to match multi-word phrases before single words
    sorted_keys = sorted(aliases.keys(), key=len, reverse=True)
    
    for key in sorted_keys:
        if key in expanded:
            expanded += " " + aliases[key]
    
    return expanded

#111 covers multi questions
def detect_question_intent(question: str) -> str:
    #Uses the LLM to semantically classify the question into one of theretrieval 
    #intents 
    #Intents: compare, missing, score, explain, lookup, general kg, class search
      
    prompt = f"""Classify this IoT privacy compliance question into exactly one intent label.
Intents:
- compare: user wants to compare manufacturers, products, or policies against each other
- missing: user asks about gaps, failures, weaknesses, what's lacking or needs improvement
- score: user wants a compliance score, rating, percentage, or overall assessment
- explain: user wants a definition, explanation, or description of a concept or requirement
- law_lookup: user asks which laws apply, what a specific regulation requires, or legal coverage
- general_kg: user asks about the knowledge graph itself (how many manufacturers, what laws exist, etc.)
- class_search: anything else — specific feature questions, policy details, data practices
Question: "{question}"
Reply with only the intent label, nothing else."""

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        label = resp.choices[0].message.content.strip().lower()
        valid = {"compare", "missing", "score", "explain", "law_lookup", "general_kg", "class_search"}
        return label if label in valid else "class_search"
    except Exception:
        return "class_search"  # safe fallback




#--#--#--!
# ── KG RETRIEVAL LAYER ──────────────────────────────────────────────────────
# These helpers run live SPARQL queries against the graph so the LLM receives
# actual ontology content (descriptions, law annotation text, triples)
def _sparql_class_descriptions(class_iri_list: list[str]) -> dict[str, str]:
    """
    For each class IRI, pull rdfs:comment + any law-annotation predicate text
    out of the graph and return {iri: combined_text}.
    """
    results = {}
    for c_iri in class_iri_list:
        node = URIRef(c_iri)
        parts = []
        for obj in g.objects(node, RDFS.comment):
            parts.append(str(obj))
        for pred in LAW_PREDICATES:
            for obj in g.objects(node, pred):
                parts.append(str(obj))
        if parts:
            results[c_iri] = " ".join(parts)[:400]   # cap per-class text
    return results


def _sparql_manufacturer_triples(manufacturer_iri: str) -> list[str]:
    """
    Pull all non-trivial triples where the manufacturer is the subject.
    Returns human-readable strings like "Ring rdf:type SecurityCamera".
    """
    inst = URIRef(manufacturer_iri)
    skip_preds = {RDF.type, RDFS.label, POLICY_PROP, USER_ADDED_PROP,
                  APPLIES_TO_STATE_PROP}
    lines = []
    for pred, obj in g.predicate_objects(inst):
        if pred in skip_preds:
            continue
        pred_name = local_name(str(pred))
        obj_name  = local_name(str(obj)) if isinstance(obj, URIRef) else str(obj)[:120]
        lines.append(f"{pred_name}: {obj_name}")
    return lines[:20]   # cap to avoid flooding the prompt


def _sparql_law_annotation_text(law_id: str) -> list[str]:
    """
    For a given law id (e.g. 'CCPA'), find all classes annotated with that
    law's keywords and return their annotation snippets (up to 6 classes).
    """
    law = next((l for l in LAWS if l["id"] == law_id), None)
    if not law:
        return []
    snippets = []
    for c_iri in class_iris:
        if law_id not in class_to_laws.get(c_iri, set()):
            continue
        node = URIRef(c_iri)
        for pred in LAW_PREDICATES:
            for obj in g.objects(node, pred):
                text = str(obj)
                segs = extract_law_segments(text)
                for seg_list in segs.values():
                    snippets.extend(seg_list[:2])
        if len(snippets) >= 12:
            break
    return snippets[:12]


def query_kg_for_question(
    question: str,
    intent: str,
    manufacturer_iri: str,
    sims,
) -> dict:
    """
    Intent-driven KG retrieval.  Returns a dict with whatever raw graph data
    is most relevant to the question so build_kg_context can include it.

    Keys returned (all optional, may be empty):
      class_descriptions  : {label: description_text}  for top similar classes
      manufacturer_triples: [string, ...]               direct triples from KG
      law_snippets        : {law_label: [snippet, ...]} annotation text per law
      related_classes     : [label, ...]                classes linked via KG
    """
    kg_data: dict = {
        "class_descriptions":   {},
        "manufacturer_triples": [],
        "law_snippets":         {},
        "related_classes":      [],
    }

    # ── Always: top-K class descriptions from the graph ──────────────────
    TOP_K_RETRIEVAL = 8
    top_idxs = sorted(
        range(len(class_iris)),
        key=lambda i: sims[i],
        reverse=True
    )[:TOP_K_RETRIEVAL]

    top_iris   = [class_iris[i]  for i in top_idxs]
    top_labels = [class_labels[i] for i in top_idxs]
    desc_map   = _sparql_class_descriptions(top_iris)
    for iri, label in zip(top_iris, top_labels):
        if iri in desc_map:
            kg_data["class_descriptions"][label] = desc_map[iri]

    # ── Always: manufacturer's own triples from KG ────────────────────────
    kg_data["manufacturer_triples"] = _sparql_manufacturer_triples(
        manufacturer_iri
    )

    # ── Intent-specific: law annotation text ─────────────────────────────
    if intent in ("law_lookup", "missing", "score", "explain"):
        # Pull annotation text for every law that has classes above threshold
        covered_law_ids = set()
        for i, score in enumerate(sims):
            if float(score) >= COVERAGE_THRESHOLD:
                covered_law_ids.update(class_to_laws.get(class_iris[i], set()))
        for lid in list(covered_law_ids)[:3]:   # cap at 3 laws
            law_obj = next((l for l in LAWS if l["id"] == lid), None)
            if law_obj:
                snippets = _sparql_law_annotation_text(lid)
                if snippets:
                    kg_data["law_snippets"][law_obj["label"]] = snippets

    # ── Intent-specific: classes directly linked in the KG (subClassOf etc) ──
    if intent in ("explain", "class_search"):
        inst = URIRef(manufacturer_iri)
        linked = set()
        # rdf:type chains
        for cls in g.objects(inst, RDF.type):
            for sub in g.subjects(RDFS.subClassOf, cls):
                linked.add(local_name(str(sub)))
            linked.add(local_name(str(cls)))
        kg_data["related_classes"] = [
            l for l in linked
            if l.lower() not in {"thing", "namedindividual", ""}
        ][:10]

    return kg_data


def build_kg_context(question: str, manufacturer: dict, sims, intent: str = "class_search") -> str:
    """
    KG-grounded context builder.
    Runs live SPARQL retrieval then merges those results with similarity
    scores so the LLM sees ontology content
    """
    name = manufacturer["name"]
    iri  = manufacturer["iri"]

    # ── Live KG retrieval ────────────────────────────────────────────────
    kg_data = query_kg_for_question(question, intent, iri, sims)

    # ── Compliance metrics (unchanged) ───────────────────────────────────
    law_coverage, _, overall = compute_law_coverage(sims, "all")
    coverage_summary = [
        f"{lc['label']}: {lc['coverage_percent']}%"
        for lc in law_coverage
    ]

    # ── Missing classes ───────────────────────────────────────────────────
    missing = compute_missing_classes(sims, "all")
    missing_summary = [
        f"{m['class_label']} ({m['law_label']})"
        for m in missing[:8]
    ]

    # ── Assemble prompt context ───────────────────────────────────────────
    sections = [f"Manufacturer: {name}"]

    # Scores
    sections.append(
        f"Overall compliance score: {overall}%\n"
        f"Coverage by law: {', '.join(coverage_summary)}"
    )

    # Missing areas
    if missing_summary:
        sections.append(
            "Missing or weak regulatory areas:\n" +
            "\n".join(f"  - {m}" for m in missing_summary)
        )

    # KG class descriptions (actual ontology content)
    if kg_data["class_descriptions"]:
        desc_lines = []
        for label, desc in kg_data["class_descriptions"].items():
            desc_lines.append(f"  [{label}]: {desc}")
        sections.append(
            "Relevant regulatory class definitions (from knowledge graph):\n" +
            "\n".join(desc_lines)
        )

    # Manufacturer triples from KG
    if kg_data["manufacturer_triples"]:
        sections.append(
            "Knowledge graph facts about this manufacturer:\n" +
            "\n".join(f"  - {t}" for t in kg_data["manufacturer_triples"])
        )

    # Law annotation snippets
    if kg_data["law_snippets"]:
        for law_label, snippets in kg_data["law_snippets"].items():
            sections.append(
                f"Regulatory text from {law_label} (from knowledge graph):\n" +
                "\n".join(f"  • {s}" for s in snippets[:5])
            )

    # KG-linked related classes
    if kg_data["related_classes"]:
        sections.append(
            "Related ontology classes (KG structure): " +
            ", ".join(kg_data["related_classes"])
        )

    return "\n\n".join(sections)


groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", "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"))

wgrader.init_grader(
    groq_client=groq_client, manufacturers=manufacturers,
    class_iris=class_iris, class_labels=class_labels,
    class_descs=class_descs, class_to_laws=class_to_laws,
    law_to_class_idxs=law_to_class_idxs, LAWS=LAWS,
    COVERAGE_THRESHOLD=COVERAGE_THRESHOLD,
    rank_classes_for_policy=rank_classes_for_policy,
    state_detector=state_detector,
)

#--#--# reads relevent raw tripels
def ask_llm(question: str, kg_context: str) -> str:
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an IoT privacy compliance assistant. "
                        "You are given structured data pulled directly from a compliance knowledge graph, "
                        "including regulatory class definitions, law annotation text, ontology triples, "
                        "and similarity-based compliance scores.\n"
                        "- Answer using the knowledge graph data provided — prefer class definitions "
                        "and regulatory text over scores alone when they are relevant\n"
                        "- Max 4 sentences\n"
                        "- Plain English, no IRIs or technical graph jargon\n"
                        "- If the KG data does not contain enough information to answer, say so clearly\n"
                    )
                },
                {
                    "role": "user",
                    "content": f"""
DATA:
{kg_context}

QUESTION:
{question}

Answer simply:
"""
                }
            ],
            temperature=0.2,
            max_tokens=200,
        )

        return response.choices[0].message.content.strip()



    except Exception as e:
        return f"Chat error: {str(e)}"

from functools import lru_cache

@lru_cache(maxsize=100)
def cached_llm(question, context):
    return ask_llm(question, context)


@app.post("/chat")
def chat():

    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    iri = (data.get("iri") or "").strip()

    if not question or not iri:
        return jsonify({"answer": "Please select a manufacturer and ask a question."}), 400

    match = next((m for m in manufacturers if m["iri"] == iri), None)
    if not match:
        return jsonify({"answer": "Manufacturer not found."}), 404

    # Get BERT sims for this manufacturer
    _, sims = rank_classes_for_policy(match["policy"], "all")
    if sims is None:
        return jsonify({"answer": "Could not analyze this manufacturer's policy."}), 500

    # Detect intent so KG retrieval is targeted to the question type
    expanded_q = expand_question_with_aliases(question)
    intent = detect_question_intent(expanded_q)

    # Build KG-grounded context (live SPARQL retrieval + similarity scores)
    kg_context = build_kg_context(expanded_q, match, sims, intent=intent)
    answer = cached_llm(question, kg_context)

#! this is the proof o fs sfource
    return jsonify({
        "answer": answer,
        "show_card": False,
        "data_source": f"inferred_merged.rdf  SWRL active: {SWRL_ACTIVE}, {len(g)} triples loaded",
        "intent": intent,
    }), 200

@app.get("/compare")
def compare():
    iri1 = request.args.get("iri1")
    iri2 = request.args.get("iri2")
    state = request.args.get("state", "all")

    m1 = next((m for m in manufacturers if m["iri"] == iri1), None)
    m2 = next((m for m in manufacturers if m["iri"] == iri2), None)

    if not m1 or not m2:
        return jsonify({"error": "One or both manufacturers not found"}), 404

    _, sims1 = rank_classes_for_policy(m1["policy"], state)
    _, sims2 = rank_classes_for_policy(m2["policy"], state)

    coverage1, _, overall1 = compute_law_coverage(sims1, state)
    coverage2, _, overall2 = compute_law_coverage(sims2, state)

    missing1 = compute_missing_classes(sims1, state)
    missing2 = compute_missing_classes(sims2, state)

    # Find classes one covers but the other doesn't
    covered1_iris = {
        class_iris[i] for i in range(len(class_iris))
        if sims1 is not None and float(sims1[i]) >= COVERAGE_THRESHOLD
    }
    covered2_iris = {
        class_iris[i] for i in range(len(class_iris))
        if sims2 is not None and float(sims2[i]) >= COVERAGE_THRESHOLD
    }

    m1_advantage = [
        class_labels[class_iris.index(iri)]
        for iri in covered1_iris - covered2_iris
    ]
    m2_advantage = [
        class_labels[class_iris.index(iri)]
        for iri in covered2_iris - covered1_iris
    ]

    return jsonify({
        "manufacturer1": {
            "name": m1["name"],
            "overall": overall1,
            "law_coverage": coverage1,
            "missing_count": len(missing1),
            "advantages_over_other": m1_advantage[:5]
        },
        "manufacturer2": {
            "name": m2["name"],
            "overall": overall2,
            "law_coverage": coverage2,
            "missing_count": len(missing2),
            "advantages_over_other": m2_advantage[:5]
        }
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)