from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z_0-9]*|\d+\.\d+|\d+|==|!=|<=|>=|\+\+|--|&&|\|\||<<|>>|[-+*/%=&|!<>^~?:;,.()[\]{}]"
)
KEYWORDS_KW = {"for", "while", "if", "else", "switch", "case", "return", "break", "continue"}
KEYWORDS_TYPE = {"int", "long", "double", "float", "char", "bool", "void", "string", "struct"}
KEYWORDS_NORMID = {
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if",
    "int", "long", "register", "return", "short", "signed", "sizeof",
    "static", "struct", "switch", "typedef", "union", "unsigned", "void",
    "volatile", "while", "include", "define", "using", "namespace", "std",
    "cin", "cout", "scanf", "printf", "main",
}


def clean_code(code, normalize_identifiers=False, remove_comments=True):
    if remove_comments:
        code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
        code = re.sub(r"//.*", " ", code)
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    if normalize_identifiers:
        def repl(match):
            tok = match.group(0)
            return tok if tok in KEYWORDS_NORMID else "ID"
        code = re.sub(r"\b[A-Za-z_][A-Za-z_0-9]*\b", repl, code)
    return re.sub(r"\s+", " ", code).strip()


def lexical_tokens(code):
    return TOKEN_RE.findall(clean_code(code, remove_comments=True))


def structure_tokens(code):
    cleaned = clean_code(code, remove_comments=True)
    raw = TOKEN_RE.findall(cleaned)
    out = []
    depth = 0
    paren = 0
    for tok in raw:
        if tok == "{":
            depth += 1
            out.append(f"BLOCK_OPEN_D{min(depth, 8)}")
        elif tok == "}":
            out.append(f"BLOCK_CLOSE_D{min(depth, 8)}")
            depth = max(0, depth - 1)
        elif tok == "(":
            paren += 1
            out.append(f"PAREN_OPEN_D{min(paren, 8)}")
        elif tok == ")":
            out.append(f"PAREN_CLOSE_D{min(paren, 8)}")
            paren = max(0, paren - 1)
        elif tok == ";":
            out.append(f"STMT_END_D{min(depth, 8)}")
        elif tok in KEYWORDS_KW:
            out.append(f"KW_{tok.upper()}_D{min(depth, 8)}")
        elif tok in KEYWORDS_TYPE:
            out.append(f"TYPE_{tok.upper()}")
        elif tok in {"+", "-", "*", "/", "%"}:
            out.append("OP_ARITH")
        elif tok in {"==", "!=", "<=", ">=", "<", ">"}:
            out.append("OP_CMP")
        elif tok in {"=", "+=", "-=", "*=", "/="}:
            out.append("OP_ASSIGN")
        elif tok in {"&&", "||", "!"}:
            out.append("OP_LOGIC")
        elif re.fullmatch(r"\d+(?:\.\d+)?", tok):
            out.append("LITERAL_NUM")
        elif re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", tok):
            out.append(f"IDENT_D{min(depth, 8)}")
    return out


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def preprocess_split(rows):
    out = []
    for row in rows:
        code = row["code"]
        lex = lexical_tokens(code)
        struct = structure_tokens(code)
        out.append({
            "index": str(row.get("id", row.get("index", ""))),
            "label": str(row["label"]),
            "lexical_text": " ".join(lex),
            "lexical_normid_text": " ".join(lexical_tokens(clean_code(code, normalize_identifiers=True))),
            "structure_text": " ".join(struct),
            "code_length": len(lex),
            "ast_proxy_node_count": len(struct),
            "ast_parse_ok": bool(struct),
        })
    return out


def dataset_statistics(rows, split_name):
    labels = {}
    lengths = []
    struct_lengths = []
    parse_ok = 0
    for r in rows:
        labels[r["label"]] = labels.get(r["label"], 0) + 1
        lengths.append(r["code_length"])
        struct_lengths.append(r["ast_proxy_node_count"])
        if r["ast_parse_ok"]:
            parse_ok += 1
    return {
        "split": split_name,
        "rows": len(rows),
        "labels": len(labels),
        "min_per_label": min(labels.values()),
        "max_per_label": max(labels.values()),
        "mean_code_tokens": float(np.mean(lengths)),
        "median_code_tokens": float(np.median(lengths)),
        "p95_code_tokens": float(np.percentile(lengths, 95)),
        "mean_ast_proxy_tokens": float(np.mean(struct_lengths)),
        "ast_proxy_parse_ok_rate": parse_ok / max(len(rows), 1),
    }


def fit_transform_tfidf(train_texts, test_texts, ngram_range, max_features, seed):
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=2,
        sublinear_tf=True,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )
    train_x = vectorizer.fit_transform(train_texts)
    test_x = vectorizer.transform(test_texts)
    n_components = min(256, max(2, min(train_x.shape[0] - 1, train_x.shape[1] - 1)))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    train_z = normalize(svd.fit_transform(train_x), norm="l2")
    test_z = normalize(svd.transform(test_x), norm="l2")
    return train_z, test_z


def concat_normalized(parts, weights):
    scaled = []
    for arr, w in zip(parts, weights):
        scaled.append(normalize(arr, norm="l2") * w)
    return normalize(np.hstack(scaled), norm="l2")


def evaluate_embeddings(embeddings, labels):
    emb = normalize(embeddings.astype(np.float32), norm="l2")
    sims = cosine_similarity(emb, emb)
    np.fill_diagonal(sims, -np.inf)
    labels_arr = np.array(labels)
    n = len(labels)
    map_scores, r1, r5, r10, rr = [], [], [], [], []
    for i in range(n):
        relevant = np.where(labels_arr == labels_arr[i])[0]
        relevant = relevant[relevant != i]
        r = len(relevant)
        if r == 0:
            continue
        order = np.argpartition(-sims[i], kth=min(r, n - 2))[:r]
        order = order[np.argsort(-sims[i][order])]
        rel_set = set(relevant.tolist())
        hits = 0
        precisions = []
        first_rank = None
        for rank, cand in enumerate(order, start=1):
            if cand in rel_set:
                hits += 1
                precisions.append(hits / rank)
                if first_rank is None:
                    first_rank = rank
        map_scores.append(sum(precisions) / r)
        top10 = order[:10]
        r1.append(int(top10[0] in rel_set))
        r5.append(int(any(c in rel_set for c in order[:5])))
        r10.append(int(any(c in rel_set for c in order[:10])))
        rr.append(0.0 if first_rank is None else 1.0 / first_rank)
    return {
        "MAP@R": float(np.mean(map_scores)),
        "Recall@1": float(np.mean(r1)),
        "Recall@5": float(np.mean(r5)),
        "Recall@10": float(np.mean(r10)),
        "MRR": float(np.mean(rr)),
    }


RUN_PLAN = [
    {"model": "tfidf_token", "scope": "full",
     "text_field": "lexical_text", "ngram": (1, 3), "max_features": 80000,
     "norm_identifiers": False, "include_structure": False, "ast_only": False,
     "lexical_weight": 1.0, "structure_weight": 0.0},
    {"model": "tfidf_token_normid", "scope": "full_normid",
     "text_field": "lexical_normid_text", "ngram": (1, 3), "max_features": 80000,
     "norm_identifiers": True, "include_structure": False, "ast_only": False,
     "lexical_weight": 1.0, "structure_weight": 0.0},
    {"model": "ast_proxy_only", "scope": "full",
     "text_field": "structure_text", "ngram": (1, 4), "max_features": 40000,
     "norm_identifiers": False, "include_structure": False, "ast_only": True,
     "lexical_weight": 0.0, "structure_weight": 1.0},
    {"model": "gac_tfidf_structure", "scope": "full_w05",
     "text_field": "lexical_text", "ngram": (1, 3), "max_features": 80000,
     "norm_identifiers": False, "include_structure": True, "ast_only": False,
     "lexical_weight": 1.0, "structure_weight": 0.5},
    {"model": "gac_tfidf_structure", "scope": "full_w09",
     "text_field": "lexical_text", "ngram": (1, 3), "max_features": 80000,
     "norm_identifiers": False, "include_structure": True, "ast_only": False,
     "lexical_weight": 1.0, "structure_weight": 0.9},
    {"model": "gac_tfidf_structure_normid", "scope": "robust_normid",
     "text_field": "lexical_normid_text", "ngram": (1, 3), "max_features": 80000,
     "norm_identifiers": True, "include_structure": True, "ast_only": False,
     "lexical_weight": 1.0, "structure_weight": 0.9},
]
SEEDS = [42, 2026, 3407]


def run_plan_item(plan, seed, train_proc, test_proc):
    np.random.seed(seed)
    if plan["ast_only"]:
        train_struct = [r["structure_text"] for r in train_proc]
        test_struct = [r["structure_text"] for r in test_proc]
        _, test_emb = fit_transform_tfidf(train_struct, test_struct,
                                          plan["ngram"], plan["max_features"], seed)
    else:
        field = plan["text_field"]
        train_texts = [r[field] for r in train_proc]
        test_texts = [r[field] for r in test_proc]
        _, test_lex = fit_transform_tfidf(train_texts, test_texts,
                                          plan["ngram"], plan["max_features"], seed)
        if plan["include_structure"]:
            train_struct = [r["structure_text"] for r in train_proc]
            test_struct = [r["structure_text"] for r in test_proc]
            _, test_struct_emb = fit_transform_tfidf(train_struct, test_struct,
                                                    (1, 4), 40000, seed)
            test_emb = concat_normalized([test_lex, test_struct_emb],
                                          [plan["lexical_weight"], plan["structure_weight"]])
        else:
            test_emb = test_lex
    labels = [r["label"] for r in test_proc]
    return evaluate_embeddings(test_emb, labels)
