"""
Production login/DB boundary hardening — sanitized persistence-failure
classification for the HTTP adapter layer (rcs.review.http_api) only.

Business logic (auth.py, sessions.py, service.py, entra_auth.py, ...)
keeps raising/propagating the real driver exceptions completely
unchanged — this module never wraps or suppresses anything upstream of
the HTTP boundary. Its only job is to answer, at the one place responses
are written, "did the configured persistence backend just fail to
reach/authenticate/authorize, as opposed to the caller's own request
being invalid?" — so http_api.py can return one stable, sanitized
`{"error": "service_unavailable"}` (503) instead of ever letting a raw
pymongo/SQLAlchemy exception (connection strings, hostnames, driver
class names, "Authentication failed.") reach the browser.

Deliberately narrow, matching rcs.review.db_retry / mongo_retry's own
discipline: this recognizes a small, explicit set of infrastructure-
failure exception types per backend. It never touches domain exceptions
(ItemNotFound, Forbidden, VersionConflict, InvalidTransition, ValueError)
-- those already have their own sanitized handling in http_api.py and
must keep reaching it unclassified here.
"""

from __future__ import annotations

from pymongo.errors import ConfigurationError as MongoConfigurationError
from pymongo.errors import ConnectionFailure, OperationFailure
from sqlalchemy.exc import OperationalError as PostgresOperationalError

from rcs.review.persistence import MissingPersistenceConfig, UnsupportedPersistenceBackend

# MongoDB server error codes meaning "the driver reached a real
# mongod/mongos, but authentication or authorization was rejected" --
# https://www.mongodb.com/docs/manual/reference/error-codes/. Deliberately
# narrow: any OTHER OperationFailure code (bad operator, duplicate key
# via DuplicateKeyError, a real write conflict, ...) is a normal
# application-level error, not an infrastructure failure, and must not
# be swallowed into the same "service unavailable" bucket.
_MONGO_AUTH_FAILURE_CODES = frozenset({18, 13})  # AuthenticationFailed, Unauthorized


def is_persistence_unavailable(exc: BaseException) -> bool:
    """True if `exc` is a recognized "the configured persistence backend
    could not be reached, or rejected the configured credentials/
    authorization" failure -- never a normal invalid-input/domain
    exception. Callers map True to one sanitized, stable response;
    False means the caller's existing (unrelated) exception handling
    should run instead."""
    if isinstance(exc, (MissingPersistenceConfig, UnsupportedPersistenceBackend)):
        return True
    if isinstance(exc, (ConnectionFailure, MongoConfigurationError)):
        return True
    if isinstance(exc, OperationFailure) and getattr(exc, "code", None) in _MONGO_AUTH_FAILURE_CODES:
        return True
    if isinstance(exc, PostgresOperationalError):
        return True
    return False
