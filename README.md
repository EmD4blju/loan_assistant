# Loan assistant

This application lets you assess your loan approval chances from home—instantly providing a clear probability of success and tailored recommendations to strengthen your application when the odds are low. Behind the scenes, it analyzes key financial and personal indicators using machine learning to highlight the factors most influencing your outcome. You receive practical, prioritized guidance so each improvement cycle meaningfully boosts your likelihood of approval.

## User's story

**Persona**

- **_Name:_** John (potential performance car buyer)<br>
- **_Goal:_** Secure financing to purchase a Toyota Supra (~$150k) without wasting time on a likely in‑person rejection.

**Primary User Story**  
As a buyer seeking financing for a high-value (but mainstream) performance car, John wants to quickly understand his approval likelihood and what he can improve so that he can make informed decisions before approaching a lender.

**Journey (Happy / Learning Path)**

1. **Discovery:** John visits LoanAssistant after a dealer pre-quote makes him unsure his current credit profile and down payment are strong enough.
2. **Onboarding:** A clear landing page explains the flow: provide basics → see approval probability → get tailored improvement plan.
3. **Questionnaire:** A guided form gathers income, employment stability, existing debt payments, credit score range, down payment amount, requested principal, and desired term. Inline tips explain how each affects approval.
4. **Processing:** The system evaluates risk factors and computes an approval probability.
5. **Result Screen:** John sees a moderate probability (e.g., “Current estimated approval likelihood: 38%”). Limiting factors are highlighted.
6. **Reaction:** Mild concern—he could apply now, but approval is uncertain.
7. **Guidance Mode:** The integrated advisor produces prioritized, actionable recommendations.
8. **Scenario Exploration:** John tweaks down payment and debt payoff assumptions; simulations recalc probability (e.g., 38% → 54% with changes).
9. **Export:** He downloads a structured report summarizing inputs, probability, limiting factors, prioritized recommendations, and scenario deltas.
10. **Next Step:** John schedules a 60‑day plan: pay down revolving debt + accumulate additional cash before submitting a formal application.

**Acceptance Criteria (Illustrative)**

- Questionnaire completion target: under 5 minutes with progress indicator and autosave.
- Result view: numeric probability + top 3 limiting factors + plain-language explanation for each.
- Guidance: at least 3 tailored recommendations tied to specific metrics.
- Scenario tool: user can adjust at least down payment, loan term, and a debt payoff slider and re-run instantly.
- Export: includes timestamp, anonymized session ID, baseline probability, scenario comparison table, and recommended next actions.

**Outcome**<br>
Instead of guessing or risking a hard inquiry with mediocre odds, John leaves with a structured improvement roadmap, a higher projected approval path, and clarity on when to formally apply.

## Dependencies

[Loan Approval Classification Data](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data) - _Ta-wei Lo_

# Application Setup & Dependency Installation

This folder contains the Streamlit application. Follow the steps below to install dependencies and run the app.

## 1. Prerequisites

- Python 3.12+ (see `requires-python` in `pyproject.toml`)
- One of the following dependency managers:
  - [astral-uv](https://github.com/astral-sh/uv) (recommended for speed & lockfile reproducibility)
  - Standard `pip`

Optional but encouraged: create an isolated virtual environment.

### Create / Activate a Virtual Environment (if using pip)

You can skip this if you rely on `uv` (it manages environments automatically).

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux / macOS
```

## 2. Install Dependencies

### Option A: Using astral-uv (preferred)

The repository already includes a `pyproject.toml` and `uv.lock`. Just sync:

```bash
uv sync
```

This will:

- Create (or reuse) a managed environment
- Install all declared dependencies (e.g., Streamlit)
- Respect the lockfile for reproducible builds

### Option B: Using pip

If you prefer plain `pip`, first generate a `requirements.txt` from the project metadata (optional), or maintain one manually. If a `requirements.txt` exists, install with:

```bash
pip install -r requirements.txt
```

## 3. Run the Application

From inside this `app/` directory (or provide the path), launch Streamlit pointing at `main.py`:

```bash
streamlit run main.py
```

After running, Streamlit will print a local URL (typically http://localhost:8501) you can open in your browser.

## 4. Updating / Adding Dependencies

- With `uv`: add normally, then `uv add package_name` and commit updated `uv.lock`.
- With `pip`: add to `requirements.txt` manually (pin versions for reproducibility) and reinstall.

## 5. Troubleshooting

| Issue                           | Possible Cause            | Fix                                                                                       |
| ------------------------------- | ------------------------- | ----------------------------------------------------------------------------------------- |
| Command `streamlit` not found   | Environment not activated | Activate venv or rerun `uv sync`                                                          |
| Version mismatch error (Python) | Python < 3.12             | Install Python 3.12+                                                                      |
| Dependencies not updating       | Cached env                | For `uv` run `uv sync --refresh`; for pip run `pip install -r requirements.txt --upgrade` |

## 6. Quick Start (Copy/Paste)

```bash
# Using uv
uv sync && streamlit run main.py

# Or using pip
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && streamlit run main.py
```

---

Feel free to extend `main.py` with Streamlit UI components. Currently it only prints a placeholder message.
