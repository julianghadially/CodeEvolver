"""POST /import-job-history optional fields and their read path.

Covers the research-run display fields (name, held-out scores, memory /
meta-reflection / integrity reports): import writes them with skip-on-None
semantics, GET /job/{id} returns all of them, and GET /jobs returns the name
and scores but not the long markdown reports.
"""
import asyncio

import pytest

import src.api.main as api_main
from src.api.auth.local_user import LOCAL_STUB_USER
from src.api.schemas.requests import ImportJobHistoryRequest


class _FakeCursor:
    def __init__(self, docs):
        self._docs = docs

    def sort(self, *args, **kwargs):
        return self

    def skip(self, n):
        return self

    def limit(self, n):
        return self

    def __aiter__(self):
        async def gen():
            for doc in self._docs:
                yield doc
        return gen()


class _FakeUpdateResult:
    def __init__(self, upserted_id):
        self.upserted_id = upserted_id


class _FakeJobs:
    """Just enough of motor's collection API for the endpoints under test."""

    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.last_update: dict | None = None

    async def find_one(self, query):
        return self.docs.get(query["job_id"])

    async def update_one(self, query, update, upsert=False):
        self.last_update = update
        job_id = query["job_id"]
        created = job_id not in self.docs
        doc = self.docs.setdefault(job_id, {"job_id": job_id})
        if created:
            doc.update(update.get("$setOnInsert", {}))
        doc.update(update["$set"])
        return _FakeUpdateResult(job_id if created else None)

    async def count_documents(self, query):
        return len(self.docs)

    def find(self, query, projection=None):
        return _FakeCursor(list(self.docs.values()))


class _FakeDB:
    def __init__(self):
        self.jobs = _FakeJobs()


@pytest.fixture
def db(monkeypatch):
    fake = _FakeDB()
    monkeypatch.setattr(api_main, "get_database", lambda: fake)
    return fake


FULL = dict(
    job_id="job_abc",
    iteration_history={"0": {"git_branch": "codeevolver-main"}},
    status="completed",
    name="ee-hover-s1",
    seed_test_score=0.58,
    final_test_score=0.92,
    memory_md="# memory",
    meta_reflection_md="# meta",
    integrity_audit_md="# integrity",
)


def _import(**kwargs):
    return asyncio.run(
        api_main.import_job_history(ImportJobHistoryRequest(**kwargs), LOCAL_STUB_USER)
    )


def test_import_sets_new_fields(db):
    resp = _import(**FULL)
    assert resp.created is True
    set_fields = db.jobs.last_update["$set"]
    for field in ("name", "seed_test_score", "final_test_score",
                  "memory_md", "meta_reflection_md", "integrity_audit_md"):
        assert set_fields[field] == FULL[field]


def test_reimport_without_fields_keeps_stored_values(db):
    _import(**FULL)
    _import(job_id="job_abc", iteration_history=FULL["iteration_history"])
    set_fields = db.jobs.last_update["$set"]
    assert "name" not in set_fields
    assert "meta_reflection_md" not in set_fields
    stored = db.jobs.docs["job_abc"]
    assert stored["name"] == "ee-hover-s1"
    assert stored["integrity_audit_md"] == "# integrity"


def test_get_job_returns_all_fields(db):
    _import(**FULL)
    job = asyncio.run(api_main.get_job_status("job_abc", LOCAL_STUB_USER))
    assert job.name == "ee-hover-s1"
    assert job.seed_test_score == 0.58
    assert job.final_test_score == 0.92
    assert job.memory_md == "# memory"
    assert job.meta_reflection_md == "# meta"
    assert job.integrity_audit_md == "# integrity"


def test_list_jobs_returns_name_and_scores_not_reports(db):
    _import(**FULL)
    listing = asyncio.run(api_main.list_jobs(0, 20, LOCAL_STUB_USER))
    (job,) = listing.jobs
    assert job.name == "ee-hover-s1"
    assert job.seed_test_score == 0.58
    assert job.final_test_score == 0.92
    assert job.memory_md is None
    assert job.meta_reflection_md is None
    assert job.integrity_audit_md is None
