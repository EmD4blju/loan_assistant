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
