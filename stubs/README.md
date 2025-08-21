# Local typing stubs (stubs/) — policy and maintenance

This directory contains narrow .pyi interface stubs for third-party libraries and dynamic modules used by this codebase. These stubs exist solely for static type-checking (mypy) and are not imported at runtime.

Key points
- mypy is configured to look for stubs here via mypy_path that includes "stubs" in [pyproject.toml](pyproject.toml:53) and mirrored in [mypy.ini](mypy.ini:1).
- Stubs should model only the minimal API surface that the repository actually uses.
- Avoid Any in stubs. Prefer precise types; when not feasible, use object as a last resort.
- Keep stubs small and easy to maintain. Expand only when mypy demands it due to new usage.

When to add or update a stub
- A third-party dependency has no types (or incomplete types) and mypy reports missing attributes.
- A module uses dynamic exports (__getattr__, star imports, runtime mutation) that defeat static analysis.
- We want to lock down a stable, narrow interface for a heavy dependency (e.g., faiss, torch) to keep strict typing practical.

Directory layout
- duckdb/ __init__.pyi
- faiss/ __init__.pyi
- simplefix/ __init__.pyi
- yfinance/ __init__.pyi
- torch/ __init__.pyi
- torch/ nn.pyi
- sklearn/ __init__.pyi
- sklearn/ cluster.pyi
- sklearn/ preprocessing.pyi

Authoring guidelines
- Use Python 3.11 typing (PEP 604 union, typing.Protocol where useful).
- Prefer concrete types over Any. For arrays use numpy.typing.NDArray where helpful.
- Keep return types specific (e.g., tuple[NDArray[np.float64], NDArray[np.int64]] rather than tuple[object, object]) when our code relies on them.
- For pandas- or duckdb-returned dataframes, you may use object if adding a hard dependency to pandas-stubs is undesirable at the stub level. The repo already uses pandas-stubs in pre-commit, so DataFrame types are acceptable if needed later.
- Mirror fluent APIs accurately (methods returning self should be annotated to return the class).
- Only include names we actually use. Do not attempt to reproduce entire libraries.
- No implementations in .pyi files—signatures only.

Workflow to create or extend a stub
1. Run mypy and note the error(s) for the library or module.
2. Inspect the usage sites to determine the minimal methods/attributes needed and their expected types.
3. Author or extend the corresponding .pyi under stubs/.
4. Re-run mypy. Iterate until errors are resolved without introducing Any.
5. Keep commit messages scoped, e.g., "stubs(faiss): add IndexIDMap.add_with_ids".

Examples of minimal stubs (illustrative)

duckdb (__init__.pyi)
class DuckDBPyConnection:
    def execute(self, sql: str, parameters: object | None = ...) -> DuckDBPyConnection: ...
    def fetchall(self) -> list[tuple[object, ...]]: ...
    def dataframe(self) -> object: ...
def connect(database: str | None = ...) -> DuckDBPyConnection: ...

faiss (__init__.pyi)
class Index:
    d: int
    is_trained: bool
    ntotal: int
    def add(self, x: object) -> None: ...
    def search(self, x: object, k: int) -> tuple[object, object]: ...
    def reset(self) -> None: ...
    def train(self, x: object) -> None: ...
class IndexFlatL2(Index):
    def __init__(self, d: int) -> None: ...
class IndexFlatIP(Index):
    def __init__(self, d: int) -> None: ...
class IndexIDMap(Index):
    def __init__(self, index: Index) -> None: ...
    def add_with_ids(self, x: object, ids: object) -> None: ...

simplefix (__init__.pyi) — suggested shape
class Message:
    def __init__(self) -> None: ...
    def append_pair(self, tag: int, value: str) -> None: ...
    def encode(self) -> bytes: ...
class FixParser:
    def __init__(self) -> None: ...
    def append(self, data: bytes) -> None: ...
    def get_message(self) -> Message | None: ...

sklearn (cluster.pyi, preprocessing.pyi) — suggested shape
class KMeans:
    def __init__(self, n_clusters: int = ..., random_state: int | None = ...) -> None: ...
    def fit(self, X: object, y: object | None = ...) -> KMeans: ...
    def fit_predict(self, X: object, y: object | None = ...) -> object: ...
class StandardScaler:
    def __init__(self, with_mean: bool = ..., with_std: bool = ...) -> None: ...
    def fit(self, X: object, y: object | None = ...) -> StandardScaler: ...
    def transform(self, X: object) -> object: ...

torch (__init__.pyi, nn.pyi) — suggested shape
class Tensor: ...
def tensor(data: object, dtype: object | None = ..., device: object | None = ...) -> Tensor: ...
def from_numpy(a: object) -> Tensor: ...
class nn_Module:
    def __init__(self) -> None: ...
    def forward(self, *args: object, **kwargs: object) -> object: ...
class Linear(nn_Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = ...) -> None: ...

Maintenance and ownership
- Keep stubs aligned with real usage. If mypy starts failing after a library upgrade, update only the necessary members.
- Prefer refactoring our code to use narrower, typed adapters rather than expanding stubs when feasible.
- Document significant stub changes in docs/development/DE-SHIMMIFICATION_PLAN.md so downstream teams know the intended contracts.

FAQ
- Q: Should we ship py.typed? A: Not necessary for local stubs; py.typed is for distributing typed packages. These stubs are in-repo and discovered via mypy_path.
- Q: Can we import pandas or numpy inside .pyi files? A: Yes for types only, but keep imports minimal to avoid accidental dependency coupling. Using object is acceptable where precision is not required.
- Q: How do we handle overloads? A: Prefer a single precise signature if possible. Use typing.overload only when its absence leads to false positives in our usage.

Checklist for new or changed stubs
- [ ] Add or update the .pyi file under stubs/
- [ ] Re-run mypy and ensure new errors are not introduced
- [ ] Link the change to a TODO item in the tracking list
- [ ] If behavior changed, capture notes in docs/development/DE-SHIMMIFICATION_PLAN.md

References
- mypy stub files: https://mypy.readthedocs.io/en/stable/stubs.html
- typing module (PEP 484+): https://docs.python.org/3/library/typing.html
- numpy.typing: https://numpy.org/devdocs/reference/typing.html

End