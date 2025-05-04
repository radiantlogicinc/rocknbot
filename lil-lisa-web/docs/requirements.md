// ... existing code ...
## 2. Authentication (Backend: `LilLisa_Server`, Frontend Interaction: `lil-lisa-web`)

### 2.1. Login Flow (Passwordless Email Link)

1.  **Login Initiation (Frontend: `lil-lisa-web`):**
    *   (As before)
2.  **Email Submission & Validation:**
    *   (As before - Frontend calls `/api/auth/request_login`)
3.  **Token Generation & Email Dispatch (Backend: `LilLisa_Server`):**
    *   If the email domain is valid, the backend generates a secure, unique, single-use, time-limited (e.g., 15 minutes) `login_token`.
    *   The backend temporarily stores a mapping from this `login_token` to the user's `email` and its expiry time. **This mapping will be stored in a global, temporary key-value store (e.g., a dedicated `speedict` file like `_global_auth_tokens/`)**, accessible without needing the `user_id_hash`.
    *   The backend sends the email containing the login link (`<frontend_url>/login_verify?token=<login_token>`).
    *   (Rest as before)
4.  **Token Verification & Session Creation:**
    *   (Frontend extracts token and calls `/api/auth/verify_token` as before)
    *   **Backend Verification (`LilLisa_Server`):**
        *   The backend API receives the `login_token`.
        *   It looks up the `login_token` in the **global temporary store**. Validates it (exists, not expired).
        *   If valid:
            *   Retrieves the associated `email` address.
            *   **Deletes the entry from the global temporary store.**
            *   Securely hashes the `email` address -> `user_id_hash`.
            *   Extracts the `domain` from the `email`.
            *   **Locates/creates the user-specific authentication `speedict` database (`<SPEEDICT_FOLDERPATH>/<AUTH_INFO_FOLDENAME>/<user_id_hash>/`).**
            *   **Stores the user's `domain` within this user-specific `speedict` DB (e.g., `{ 'domain': user_domain }`).**
            *   Generates a persistent JWT associated with the `user_id_hash`.
            *   Returns the persistent JWT and user info (`user_id_hash`, `domain`) to the frontend.
        *   If invalid/not found, returns an error.
    *   (Frontend handling as before)
5.  **Auto-Login (Persistent Session):**
    *   (Frontend checks localStorage, calls `/api/auth/verify_session` as before)
    *   **Backend Validation (`LilLisa_Server`):**
        *   Validates the received JWT (signature, expiry), extracts `user_id_hash`.
        *   **Locates the user-specific auth `speedict` DB (`.../<user_id_hash>/`).**
        *   **Attempts to load the `domain` from this DB.** If successful, the user is considered valid.
        *   Returns user info (`user_id_hash`, `domain`). If the user DB/domain info doesn't exist, returns an error.
    *   (Frontend update as before)
6.  **Logout:**
    *   (As before)

### 2.2. Security Considerations (Backend: `LilLisa_Server`)

*   (As before, JWT security is still key)
*   **Secure the global temporary token store if it's file-based.**
*   **Implement regular cleanup of expired tokens in the global temporary store.**

## 3. Shared Conversation History (Backend: `LilLisa_Server`, Frontend Interaction: `lil-lisa-web`)

### 3.1. Storage (Backend: `LilLisa_Server`)

1.  **Data Model & Structure:**
    *   **Storage Mechanism:** Conversation history metadata stored using `speedict`.
    *   **Folder Structure:** `<SPEEDICT_FOLDERPATH>/<CONVERSATION_HISTORY_FOLDENAME>/<domain>/<user_id_hash>/` (User's history DB).
    *   **Data Format:** Within user's history DB, a list under key `'history'`, each item: `{ "session_id": ..., "user_id": ..., "title": ..., "first_query": ..., "created_at": ... }`.
    *   **Storage Trigger:** (As before - `/invoke` call saves to user's history DB).
2.  **Persistence:**
    *   History data persistence via `speedict` file storage.
    *   **Authentication Info Persistence:** User domain info stored in user-specific auth `speedict` DB (`<SPEEDICT_FOLDERPATH>/<AUTH_INFO_FOLDENAME>/<user_id_hash>/`). Temporary login tokens stored in a global temporary store (e.g., separate `speedict` file).

// ... (Rest of Section 3.2 UI remains the same) ...

### 3.3. Backend API (`LilLisa_Server`)

1.  **New Auth Endpoints:**
    *   `/api/auth/request_login` (POST): Takes email, validates domain, generates `login_token`, **stores token->email mapping in global temporary store**, sends email.
    *   `/api/auth/verify_token` (GET): Takes `login_token`, **validates against global temporary store**, retrieves email, **deletes temp token entry**, hashes email -> `user_id_hash`, extracts domain, **creates/updates user-specific auth `speedict` DB with domain info**, generates/returns persistent JWT.
    *   `/api/auth/verify_session` (POST): Takes JWT, validates signature, extracts `user_id_hash`, **verifies user existence and retrieves domain by checking user-specific auth `speedict` DB**, returns user info.
    *   `/api/auth/logout` (POST): (Optional) (As before).
2.  **`/invoke` / `/invoke_stream` (POST - Modification):**
    *   Requires JWT.
    *   Verifies JWT via `/api/auth/verify_session` logic (or shared auth dependency) to get `user_id_hash`, `domain`.
    *   If authenticated and first query for `session_id`, **locates user's history `speedict` DB**, loads/appends/saves history entry.
    *   (Rest as before).
3.  **New `/api/history` (GET):**
    *   Requires JWT.
    *   Verifies JWT to get `user_id_hash`, `domain`.
    *   Accepts params (as before).
    *   **Retrieval Logic:** (As before - reads from user-specific history DB or iterates domain's user history DBs).
    *   **In-Memory Processing:** (As before).
    *   Returns JSON list (as before).

// ... (Section 4 UI/UX remains the same) ...

## 5. Configuration

*   **`lil-lisa-web`:**
    *   `LIL_LISA_SERVER_URL`: (As before).
*   **`LilLisa_Server`:**
    *   `JWT_SECRET_KEY`, `JWT_ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES`: (As before).
    *   `SPEEDICT_FOLDERPATH`: Base path for `speedict` data (existing).
    *   `AUTH_INFO_FOLDENAME`: Subfolder name within `SPEEDICT_FOLDERPATH` for auth data (new).
    *   `CONVERSATION_HISTORY_FOLDENAME`: Subfolder name within `SPEEDICT_FOLDERPATH` for history data (existing/renamed).
    *   `MAIL_SERVER`, etc.: (As before).
    *   `ALLOWED_EMAIL_DOMAINS` / `BLOCKED_EMAIL_DOMAINS`: (As before).
    *   `EMAIL_HASH_SALT`: (As before).
    *   `LOGIN_TOKEN_EXPIRE_MINUTES`: (As before).
    *   **Remove:** `SQLALCHEMY_DATABASE_URI` (No longer needed for auth).
