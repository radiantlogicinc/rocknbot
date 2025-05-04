# LilLisa_Server Backend Implementation Plan

This plan outlines the steps to implement authentication and persistent conversation history features in the `LilLisa_Server` FastAPI backend, using `speedict` for storage.

**Phase 1: Core Setup & Utilities**

1.  **Dependencies:**
    *   Add `python-jose[cryptography]` or `PyJWT` for JWT handling.
    *   Add `passlib[bcrypt]` for secure hashing.
    *   Add an email library if needed (e.g., `fastapi-mail`). Check if `smtplib` suffices.
    *   Verify `python-dotenv` and `speedict` are present.
    *   Remove `SQLAlchemy` or related DB drivers if they are truly no longer needed for any other purpose (double-check if `lancedb` or other components rely on it indirectly).
2.  **Configuration (`LilLisa_Server/.env` & `src/utils.py` or equivalent):**
    *   Add new required environment variables:
        *   `AUTH_INFO_FOLDENAME` (e.g., `auth_info`)
        *   `CONVERSATION_HISTORY_FOLDENAME` (e.g., `history`)
        *   `GLOBAL_AUTH_TOKENS_FILENAME` (e.g., `_global_auth_tokens`)
        *   `JWT_SECRET_KEY` (generate a strong secret)
        *   `JWT_ALGORITHM` (e.g., `HS256`)
        *   `ACCESS_TOKEN_EXPIRE_MINUTES` (e.g., `43200` for 30 days)
        *   `LOGIN_TOKEN_EXPIRE_MINUTES` (e.g., `15`)
        *   `EMAIL_HASH_SALT` (generate a strong secret)
        *   `MAIL_...` variables if not already present.
        *   `ALLOWED_EMAIL_DOMAINS` or `BLOCKED_EMAIL_DOMAINS`.
    *   Update config loading logic (`utils.LILLISA_SERVER_ENV_DICT` or startup logic in `main.py::lifespan`) to load and validate these new variables.
3.  **Hashing Utility (`src/auth_utils.py` - new file):**
    *   Create `def hash_email(email: str) -> str:`: Uses `hashlib.sha256` with the configured `EMAIL_HASH_SALT`.
    *   Create `def verify_email_hash(email: str, hash_to_verify: str) -> bool:` (Potentially useful for debugging/testing, not strictly needed for core flow).
4.  **JWT Utility (`src/auth_utils.py`):**
    *   Create `def create_access_token(data: dict, expires_delta: timedelta = None) -> str:`: Encodes data (e.g., `{ "sub": user_id_hash }`) into a JWT using `JWT_SECRET_KEY`, `JWT_ALGORITHM`, and calculated expiry (`ACCESS_TOKEN_EXPIRE_MINUTES`).
    *   Create `def decode_access_token(token: str) -> dict | None:`: Decodes JWT, verifies signature and expiry, returns payload or raises/returns None on error.
5.  **`speedict` Path/Helper Utilities (`src/storage_utils.py` - new file or add to `utils.py`):**
    *   Create `def get_global_token_store_path() -> str:`: Returns full path based on config.
    *   Create `def get_user_auth_store_path(user_id_hash: str) -> str:`: Returns full path based on config.
    *   Create `def get_user_history_store_path(domain: str, user_id_hash: str) -> str:`: Returns full path based on config.
    *   Create `def load_speedict_data(path: str, key: str, default: Any = None) -> Any:`: Handles opening `Rdict(path)`, getting `db[key]`, closing, returning data or default, managing errors and folder creation (`os.makedirs(os.path.dirname(path), exist_ok=True)`).
    *   Create `def save_speedict_data(path: str, key: str, data: Any):`: Handles opening `Rdict(path)`, setting `db[key] = data`, closing, managing errors and folder creation.
    *   Modify `src/lillisa_server_context.py::get_db_folderpath` to **use `get_user_history_store_path` instead** (requires passing domain and hash). *This might require refactoring how context is loaded/created.*
6.  **Expired Token Cleanup (`src/background_tasks.py` - new file or add to `main.py`):**
    *   Create `def cleanup_expired_login_tokens():`
        *   Load data from `get_global_token_store_path()` (e.g., `_global_auth_tokens`).
        *   Iterate through tokens, check `expires_at` against `datetime.utcnow()`.
        *   Remove expired entries.
        *   Save updated data back.
    *   Integrate this task to run periodically (e.g., using `FastAPI BackgroundTasks`, `apscheduler`, or a simple scheduled job if running persistently).

**Phase 2: Authentication API Endpoints (`src/auth_router.py` - new file)**

1.  **Create FastAPI Router:** `auth_router = APIRouter(prefix="/api/auth", tags=["Authentication"])`
2.  **Auth Dependency (`src/auth_utils.py`):**
    *   Create `async def get_current_user(token: str = Depends(OAuth2PasswordBearer(tokenUrl="/api/auth/token"))) -> dict:` (Note: `tokenUrl` is illustrative, might not need full OAuth2 flow here if just validating bearer token).
        *   Decode `token` using `decode_access_token`.
        *   Extract `user_id_hash` from payload (`sub`).
        *   Load user info (`domain`) from `get_user_auth_store_path(user_id_hash)` using `load_speedict_data`.
        *   If token invalid or user store not found, raise `HTTPException(status.HTTP_401_UNAUTHORIZED)`.
        *   Return `{ "user_id_hash": ..., "domain": ... }`.
3.  **Endpoint: Request Login (`POST /request_login`)**
    *   Input: Pydantic model with `email: EmailStr`.
    *   Logic:
        *   Validate email domain against configured `ALLOWED/BLOCKED_EMAIL_DOMAINS`.
        *   Generate secure random `login_token` (`secrets.token_urlsafe`).
        *   Calculate `expires_at` (`datetime.utcnow() + timedelta(minutes=LOGIN_TOKEN_EXPIRE_MINUTES)`).
        *   Load global token store data using `load_speedict_data`.
        *   Add `{ login_token: { "email": email, "expires_at": expires_at.isoformat() } }` to the data.
        *   Save data back using `save_speedict_data`.
        *   Construct frontend verification URL: `<FRONTEND_URL>/login_verify?token={login_token}` (Need `FRONTEND_URL` config).
        *   Send email using configured mail service containing the link.
    *   Output: Success message.
4.  **Endpoint: Verify Token (`GET /verify_token`)**
    *   Input: `token: str` query parameter.
    *   Logic:
        *   Load global token store data.
        *   Check if `token` exists and `expires_at` is valid. Raise `HTTPException(400)` if invalid/expired.
        *   Retrieve `email` associated with the token.
        *   Delete the `token` entry from the global store data and save it back.
        *   Hash `email` -> `user_id_hash`.
        *   Extract `domain`.
        *   Get user auth store path using `get_user_auth_store_path`.
        *   Save `{ 'user_info': { 'domain': domain } }` using `save_speedict_data` (creates/overwrites).
        *   Create persistent JWT using `create_access_token({ "sub": user_id_hash })`.
    *   Output: Pydantic model `{ access_token: str, token_type: str, user_id_hash: str, domain: str }`.
5.  **Endpoint: Verify Session (`POST /verify_session`)**
    *   Input: Depends on `get_current_user` to validate JWT from header.
    *   Logic: `get_current_user` handles verification.
    *   Output: Pydantic model `{ user_id_hash: str, domain: str }` (returned by `get_current_user`).
6.  **Endpoint: Logout (`POST /logout`) - Optional**
    *   Input: Depends on `get_current_user` (or just takes token).
    *   Logic: If implementing server-side invalidation (e.g., blacklist), add JWT ID (jti) to a blacklist store (could be another `speedict` file).
    *   Output: Success message.
7.  **Register Router:** Add `app.include_router(auth_router)` in `main.py`.

**Phase 3: History API & Logic (`src/history_router.py` - new file, modify `main.py`)**

1.  **Modify `POST /invoke` / `POST /invoke_stream` (`main.py`):**
    *   Add `user: dict = Depends(get_current_user, use_cache=False)` to the function signature (make it optional if unauthenticated access is still allowed: `user: Optional[dict] = Depends(get_current_user, use_cache=False)`).
    *   Inside the `try` block, *before* returning the response:
        *   Check if `user` is not None (i.e., authenticated).
        *   If authenticated:
            *   Get user history path: `history_path = get_user_history_store_path(user["domain"], user["user_id_hash"])`.
            *   Load current history list: `history_list = load_speedict_data(history_path, 'history', default=[])`.
            *   Check if `session_id` already exists in `history_list` (e.g., `any(item['session_id'] == session_id for item in history_list)`).
            *   If `session_id` is *not* found:
                *   Generate `title` from `nl_query` (e.g., `nl_query[:60]`).
                *   Create `new_entry = { "session_id": session_id, "user_id": user["user_id_hash"], "title": title, "first_query": nl_query, "created_at": datetime.utcnow().isoformat() }`.
                *   Append `new_entry` to `history_list`.
                *   Save updated `history_list` back using `save_speedict_data(history_path, 'history', history_list)`.
2.  **Create History Router:** `history_router = APIRouter(prefix="/api/history", tags=["History"])`
3.  **Endpoint: Get History (`GET /`)**
    *   Input: Depends on `get_current_user`. Query params `offset: int = 0`, `limit: int = 25`, `filter: str = 'domain'`, `search: Optional[str] = None`.
    *   Logic:
        *   Get `user_id_hash` and `domain` from `user` (dependency).
        *   `all_history_items = []`.
        *   If `filter == 'mine'`:
            *   `history_path = get_user_history_store_path(domain, user_id_hash)`.
            *   `all_history_items = load_speedict_data(history_path, 'history', default=[])`.
        *   Else (`filter == 'domain'`):
            *   Construct domain history path: `domain_path = os.path.join(config.SPEEDICT_FOLDERPATH, config.CONVERSATION_HISTORY_FOLDENAME, domain)`.
            *   List subdirectories (user hashes) in `domain_path` (`os.listdir`, check `os.path.isdir`).
            *   For each `user_hash_dir`:
                *   `user_history_path = os.path.join(domain_path, user_hash_dir)`.
                *   `user_items = load_speedict_data(user_history_path, 'history', default=[])`.
                *   `all_history_items.extend(user_items)`.
        *   Filter in memory:
            *   If `search`:
                *   `filtered_items = [item for item in all_history_items if search.lower() in item.get('title', '').lower()]`
            *   Else: `filtered_items = all_history_items`
        *   Sort in memory:
            *   `sorted_items = sorted(filtered_items, key=lambda item: item.get('created_at', ''), reverse=True)`
        *   Paginate:
            *   `paginated_items = sorted_items[offset : offset + limit]`
    *   Output: Return `paginated_items` (List of history item dictionaries).
4.  **Register Router:** Add `app.include_router(history_router)` in `main.py`.

**Phase 4: Refinement & Testing**

1.  **Testing:** Use FastAPI test client to test all new auth and history endpoints with various scenarios (valid/invalid tokens, new/existing users, filtering, searching, pagination).
2.  **Error Handling:** Ensure appropriate `HTTPException`s are raised for auth failures, invalid input, file I/O errors during `speedict` access.
3.  **Security Review:** Double-check JWT handling, input validation, path construction to prevent traversal issues, permissions on `speedict` folders.
4.  **Concurrency:** Be mindful that `speedict` file access isn't inherently thread-safe across multiple processes. If running multiple Uvicorn workers, file corruption *could* occur without external locking. Consider implications or use a single worker if this becomes an issue.