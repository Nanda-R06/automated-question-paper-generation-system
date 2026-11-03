import tkinter as tk
from tkinter import scrolledtext, messagebox
import re
from collections import Counter
import math

from dataset_generator import generate_questions_from_input

# =============================================================================
# Similarity metric: BLEU-1 + ROUGE-L combined
# Much more appropriate than strict BLEU for comparing exam questions,
# since two questions on the same topic can be phrased very differently.
# =============================================================================

# Common words to ignore when measuring topic overlap
_STOPWORDS = {
    'a','an','the','is','are','was','were','be','been','being',
    'have','has','had','do','does','did','will','would','could','should',
    'may','might','shall','can','need','dare','ought','used',
    'i','you','he','she','it','we','they','me','him','her','us','them',
    'my','your','his','its','our','their','this','that','these','those',
    'what','which','who','whom','whose','when','where','why','how',
    'and','but','or','nor','for','so','yet','both','either','neither',
    'not','no','only','own','same','than','too','very','just',
    'in','on','at','by','from','to','of','with','about','as','into',
    'through','during','before','after','above','below','between',
    'each','few','more','most','other','some','such','any','all',
    'if','then','because','while','although','since','unless','until',
    'whether','given','show','explain','describe','define','list',
    'state','name','identify','recall','write','state','outline',
    'discuss','compare','contrast','examine','analyze','analyse',
    'evaluate','assess','justify','argue','defend','critique','judge',
    'design','create','develop','propose','construct','formulate',
    'devise','compose','invent','plan','produce','generate',
    'use','apply','implement','demonstrate','solve','predict',
    'trace','illustrate','calculate','perform','compute',
}

def _tokenize(text):
    """Lowercase word tokens, removing stopwords for content comparison."""
    return re.findall(r'\b[a-z]+\b', text.lower())

def _content_tokens(text):
    """Tokens with stopwords removed — used for topic/keyword comparison."""
    return [w for w in _tokenize(text) if w not in _STOPWORDS and len(w) > 2]

def _lcs_length(a, b):
    """Length of longest common subsequence of two token lists."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Space-efficient LCS using two rows
    prev = [0] * (n + 1)
    for i in range(m):
        curr = [0] * (n + 1)
        for j in range(n):
            if a[i] == b[j]:
                curr[j+1] = prev[j] + 1
            else:
                curr[j+1] = max(curr[j], prev[j+1])
        prev = curr
    return prev[n]

def _rouge_l_sentence(hyp_tokens, ref_tokens):
    """ROUGE-L F1 for one hypothesis vs one reference."""
    if not hyp_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    p = lcs / len(hyp_tokens)
    r = lcs / len(ref_tokens)
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

def _unigram_f1(hyp_tokens, ref_tokens):
    """Token-overlap F1 (order-independent) between one hyp and one ref."""
    hyp_c = Counter(hyp_tokens)
    ref_c = Counter(ref_tokens)
    common = sum((hyp_c & ref_c).values())
    if common == 0:
        return 0.0
    p = common / sum(hyp_c.values())
    r = common / sum(ref_c.values())
    return 2 * p * r / (p + r)

def compute_question_similarity(hypotheses, references):
    """
    Compute a composite similarity score between two sets of questions.

    Components (equal weight):
      1. BLEU-1  — unigram precision with brevity penalty (surface overlap)
      2. ROUGE-L — best-match longest common subsequence (structural overlap)
      3. Unigram F1 on content words — topic/keyword overlap

    Returns percentage 0–100.
    """
    if not hypotheses or not references:
        return 0.0

    hyp_toks     = [_tokenize(h)         for h in hypotheses]
    ref_toks     = [_tokenize(r)         for r in references]
    hyp_content  = [_content_tokens(h)   for h in hypotheses]
    ref_content  = [_content_tokens(r)   for r in references]

    # ── 1. BLEU-1 (unigram precision + brevity penalty) ──────────────────────
    total_match = total_hyp = 0
    for ht in hyp_toks:
        hyp_uni = Counter(ht)
        # clipped count against best reference
        best_ref_uni = Counter()
        for rt in ref_toks:
            rc = Counter(rt)
            for ng in hyp_uni:
                best_ref_uni[ng] = max(best_ref_uni[ng], rc[ng])
        total_match += sum(min(cnt, best_ref_uni[w]) for w, cnt in hyp_uni.items())
        total_hyp   += sum(hyp_uni.values())
    bleu1_precision = total_match / total_hyp if total_hyp else 0.0
    hyp_len = sum(len(h) for h in hyp_toks)
    ref_len = min((abs(len(r) - hyp_len), len(r)) for r in ref_toks)[1]
    bp      = 1.0 if hyp_len >= ref_len else (math.exp(1 - ref_len/hyp_len) if hyp_len else 0.0)
    bleu1   = bp * bleu1_precision

    # ── 2. ROUGE-L: each hyp matched to its best reference ───────────────────
    rouge_scores = []
    for ht in hyp_toks:
        best = max((_rouge_l_sentence(ht, rt) for rt in ref_toks), default=0.0)
        rouge_scores.append(best)
    rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    # ── 3. Content-word F1 (topic overlap, stopwords removed) ────────────────
    f1_scores = []
    for hc in hyp_content:
        best = max((_unigram_f1(hc, rc) for rc in ref_content), default=0.0)
        f1_scores.append(best)
    content_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    # ── Composite: equal-weighted average of all three ───────────────────────
    composite = (bleu1 + rouge_l + content_f1) / 3.0
    return composite * 100   # return as percentage

# =============================================================================
# Parser — handles ANY ChatGPT format
# =============================================================================
BLOOM_LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

# Every keyword that maps to a Bloom level
_LEVEL_MAP = {
    "remember": "Remember",   "knowledge": "Remember",   "recall": "Remember",
    "understand": "Understand", "comprehension": "Understand", "understanding": "Understand",
    "apply": "Apply",          "application": "Apply",    "applying": "Apply",
    "analyze": "Analyze",      "analysis": "Analyze",     "analyse": "Analyze",
    "evaluate": "Evaluate",    "evaluation": "Evaluate",  "evaluating": "Evaluate",
    "create": "Create",        "synthesis": "Create",     "creating": "Create",
}

def _is_level_header(line):
    """
    Return canonical Bloom level name if this line is a section header.
    Handles: Remember (3 Questions), ## Remember, **Understand**, APPLY:
    NEVER matches numbered lines like: 2. Analyze the benefits...
    """
    stripped = line.strip()
    # Rule 1: digit-starting lines are always questions, never headers
    if re.match(r'^\d', stripped):
        return None
    # Strip markdown
    clean = re.sub(r'[#*_`]', '', stripped).strip().rstrip(':').strip()
    # Rule 2: headers are short (questions are long)
    if len(clean.split()) > 6:
        return None
    first_word = re.split(r'[\s\(\-]', clean)[0].lower()
    return _LEVEL_MAP.get(first_word)

def parse_gpt_questions(text):
    """
    Robustly parse ChatGPT output into:
      - by_level: dict {level: [question, ...]}
      - flat:     list of all questions

    Works for ALL of these formats:
      Format A - Level headers + restarting numbers (ChatGPT default):
        Remember (3 Questions)
        1. What is X?

      Format B - Continuous numbering 1-18:
        1. What is X?
        ...
        18. Design a Y.

      Format C - Bold/markdown headers:
        **Remember**
        1. What is X?
    """
    # Normalise line endings — critical for Windows pastes
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    by_level = {lvl: [] for lvl in BLOOM_LEVELS}
    flat = []
    current_level = None

    for raw_line in text.split('\n'):
        line = raw_line.strip()
        if not line:
            continue

        # Check for level header FIRST
        detected = _is_level_header(line)
        if detected:
            current_level = detected
            continue

        # Check for a numbered question line
        # Matches: "1. text", "1) text", "1 . text"
        m = re.match(r'^\d+\s*[\.\)]\s+(.+)', line)
        if m:
            q = m.group(1).strip()
            # Remove any trailing markup
            q = re.sub(r'\s*\*+$', '', q).strip()
            if len(q) > 5:   # minimum question length
                flat.append(q)
                if current_level:
                    by_level[current_level].append(q)

    return by_level, flat

def _bleu_color(pct):
    if pct >= 50:  return "#1a7a1a"   # green  — good similarity
    if pct >= 25:  return "#b36b00"   # amber  — moderate
    return "#cc0000"                   # red    — low

# =============================================================================
# App state — stored at module level, updated by generate_questions()
# =============================================================================
app_questions_by_level = {lvl: [] for lvl in BLOOM_LEVELS}

# =============================================================================
# Callbacks
# =============================================================================
def do_generate():
    global app_questions_by_level

    topic_text = input_text.get("1.0", tk.END).strip()
    if not topic_text:
        messagebox.showwarning("Input Required", "Please enter some topic text.")
        return

    counts = {}
    for lvl in BLOOM_LEVELS:
        if bloom_vars[lvl].get() == 1:
            try:
                n = int(bloom_counts[lvl].get())
                if n > 0:
                    counts[lvl] = n
            except Exception:
                messagebox.showwarning("Error", f"Invalid number for {lvl}")
                return

    if not counts:
        messagebox.showwarning("Error", "Select at least one Bloom level.")
        return

    result = generate_questions_from_input(topic_text, counts)

    # Save per-level
    app_questions_by_level = {lvl: [] for lvl in BLOOM_LEVELS}
    for lvl, qs in result.items():
        app_questions_by_level[lvl] = list(qs)

    # Display
    output_text.config(state="normal")
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, "APP GENERATED QUESTIONS\n")
    output_text.insert(tk.END, "-" * 60 + "\n")
    qno = 1
    for lvl, qs in result.items():
        output_text.insert(tk.END, f"\n{lvl} Level Questions:\n")
        for q in qs:
            output_text.insert(tk.END, f"{qno}. {q}\n")
            qno += 1

    # Reset BLEU display
    bleu_overall_var.set("Overall BLEU Score: —")
    bleu_overall_label.config(fg="#555555")


def do_clear():
    global app_questions_by_level
    app_questions_by_level = {lvl: [] for lvl in BLOOM_LEVELS}
    output_text.config(state="normal")
    output_text.delete("1.0", tk.END)
    gpt_text.delete("1.0", tk.END)
    bleu_overall_var.set("Overall BLEU Score: —")
    bleu_overall_label.config(fg="#555555")


def do_bleu():
    # 1. Get ChatGPT text
    gpt_raw = gpt_text.get("1.0", tk.END)

    # 2. Validate GPT input
    if not gpt_raw.strip():
        messagebox.showwarning(
            "No ChatGPT Questions",
            "Please paste ChatGPT questions into the yellow box first."
        )
        return

    # 3. Validate app output exists
    app_flat = [q for lvl in BLOOM_LEVELS for q in app_questions_by_level[lvl]]
    if not app_flat:
        messagebox.showwarning(
            "No App Questions",
            "Please click 'Generate Questions' first to generate app questions."
        )
        return

    # 4. Parse GPT questions
    gpt_by_level, gpt_flat = parse_gpt_questions(gpt_raw)

    # 5. Show diagnostic if parse failed
    if not gpt_flat:
        lines = [l.strip() for l in gpt_raw.replace('\r','').split('\n') if l.strip()]
        sample = '\n'.join(lines[:5])
        messagebox.showwarning(
            "Could Not Parse ChatGPT Questions",
            f"No numbered questions found in the pasted text.\n\n"
            f"First 5 lines received:\n{sample}\n\n"
            f"Each question must start with a number:\n"
            f"  1. What is cloud computing?\n"
            f"  2. Define IaaS."
        )
        return

    # 6. Compute similarity score
    score = compute_question_similarity(app_flat, gpt_flat)
    bleu_overall_var.set(
        f"Similarity Score: {score:.1f}%   "
    )
    bleu_overall_label.config(fg=_bleu_color(score))




# =============================================================================
# GUI
# =============================================================================
root = tk.Tk()
root.title("Bloom's Taxonomy Question Generator + BLEU Scorer")
root.geometry("1180x1060")
root.configure(bg="#F0F0F0")

TITLE_FONT  = ("Arial", 16, "bold")
HEADER_FONT = ("Arial", 11, "bold")
LABEL_FONT  = ("Arial", 10)

# Scrollable canvas wrapper
canvas = tk.Canvas(root, bg="#F0F0F0", highlightthickness=0)
vsb    = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=vsb.set)
vsb.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

mf = tk.Frame(canvas, bg="#F0F0F0")          # main frame
cw = canvas.create_window((0,0), window=mf, anchor="nw")
mf.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.bind("<Configure>", lambda e: canvas.itemconfig(cw, width=e.width))
root.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

# ── Title ─────────────────────────────────────────────────────────────────────
tk.Label(mf, text="Bloom's Taxonomy Question Generator",
         font=TITLE_FONT, bg="#F0F0F0").pack(pady=(14,4))

# ── Topic input ───────────────────────────────────────────────────────────────
tk.Label(mf, text="Enter Input Text:", font=HEADER_FONT, bg="#F0F0F0").pack(anchor="w", padx=20)
input_text = scrolledtext.ScrolledText(mf, height=5, font=LABEL_FONT, relief="solid", bd=1)
input_text.pack(fill="x", padx=20, pady=(2,8))

# ── Bloom level selectors ─────────────────────────────────────────────────────
sf = tk.LabelFrame(mf, text="Select Bloom Levels", font=HEADER_FONT, bg="#F0F0F0")
sf.pack(fill="x", padx=20, pady=6)
bloom_vars = {}; bloom_counts = {}
for i, lvl in enumerate(BLOOM_LEVELS):
    v = tk.IntVar(value=1)
    tk.Checkbutton(sf, text=lvl, variable=v, font=LABEL_FONT, bg="#F0F0F0"
                   ).grid(row=i, column=0, sticky="w", padx=10)
    tk.Label(sf, text="Questions:", font=LABEL_FONT, bg="#F0F0F0").grid(row=i, column=1)
    e = tk.Entry(sf, width=4, font=LABEL_FONT)
    e.insert(0, "3"); e.grid(row=i, column=2, padx=6)
    bloom_vars[lvl] = v; bloom_counts[lvl] = e

# ── Action buttons ────────────────────────────────────────────────────────────
bf = tk.Frame(mf, bg="#F0F0F0"); bf.pack(pady=8)
tk.Button(bf, text="Generate Questions", width=22, bg="#2874A6", fg="white",
          font=HEADER_FONT, command=do_generate).pack(side="left", padx=8)
tk.Button(bf, text="Clear All", width=12, bg="#717D7E", fg="white",
          font=LABEL_FONT, command=do_clear).pack(side="left", padx=8)

# ── App output ────────────────────────────────────────────────────────────────
tk.Label(mf, text="App Generated Questions:", font=HEADER_FONT,
         bg="#F0F0F0").pack(anchor="w", padx=20)
output_text = scrolledtext.ScrolledText(mf, height=11, font=LABEL_FONT, relief="solid", bd=1)
output_text.pack(fill="x", padx=20, pady=(2,8))

# ── Divider ───────────────────────────────────────────────────────────────────
tk.Frame(mf, height=2, bg="#2874A6").pack(fill="x", padx=20, pady=6)

# ── ChatGPT paste box ─────────────────────────────────────────────────────────
tk.Label(mf, text="Paste ChatGPT Generated Questions:",
         font=HEADER_FONT, bg="#F0F0F0").pack(anchor="w", padx=20)
tk.Label(mf,
         text="Paste exactly as ChatGPT gives it — level headers + questions or plain numbered list",
         font=("Arial", 9), fg="#555555", bg="#F0F0F0").pack(anchor="w", padx=22)
gpt_text = scrolledtext.ScrolledText(mf, height=11, font=LABEL_FONT,
                                      bg="#FEFEF0", relief="solid", bd=1)
gpt_text.pack(fill="x", padx=20, pady=(2,8))

# ── BLEU button ───────────────────────────────────────────────────────────────
tk.Button(mf, text="Calculate BLEU Score", width=26, bg="#1E8449", fg="white",
          font=HEADER_FONT, command=do_bleu).pack(pady=8)

# ── Results panel ─────────────────────────────────────────────────────────────
rf = tk.LabelFrame(mf, text="BLEU Score Results", font=HEADER_FONT, bg="#F0F0F0")
rf.pack(fill="x", padx=20, pady=8)

bleu_overall_var = tk.StringVar(value="Similarity Score: —")
bleu_overall_label = tk.Label(rf, textvariable=bleu_overall_var,
                               font=("Arial", 15, "bold"), bg="#F0F0F0", fg="#555555")
bleu_overall_label.pack(pady=16, padx=20, anchor="w")

tk.Frame(mf, height=20, bg="#F0F0F0").pack()
root.mainloop()