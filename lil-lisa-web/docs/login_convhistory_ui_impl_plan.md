# lil-lisa-web Frontend Implementation Plan

This plan outlines the steps to implement the frontend UI and logic for authentication and conversation history features in the `lil-lisa-web` Flask application, interacting with the `LilLisa_Server` backend.

**Assumptions:**

*   The backend (`LilLisa_Server`) API endpoints (`/api/auth/*`, `/api/history`, modified `/invoke`) are implemented according to the backend plan.
*   The necessary `LIL_LISA_SERVER_URL` environment variable is configured for `lil-lisa-web`.
*   HTML, CSS, and JS remain embedded within `lil-lisa-web/main.py` for this phase.

**Phase 1: Authentication UI & Core Logic**

1.  **HTML Structure (`main.py::index` return string):**
    *   **Header:** Add a `Login` button (`id="login-btn"`). Add a `Logout` button (`id="logout-btn"`, initially hidden `style="display: none;"`). Add a placeholder for user info (`id="user-info"`, initially empty, hidden).
    *   **Login Modal/Section:** Create a hidden section or modal (`id="login-modal"`) containing:
        *   An email input field (`id="login-email"`, `type="email"`).
        *   A submit button (`id="login-submit-btn"`).
        *   A close button/link.
        *   Placeholders for messages (`id="login-message"`).
    *   **Body Class:** Add a class to the `<body>` tag, e.g., `logged-out`, which will be toggled.
2.  **CSS (`main.py::index` style block):**
    *   Style the Login/Logout buttons and user info display.
    *   Style the login modal/section (positioning, appearance).
    *   Add rules based on `body.logged-out` and `body.logged-in` classes to control visibility of Login vs. Logout/User Info/History Panel.
3.  **JavaScript Core (`main.py::index` script block):**
    *   **Global Variables/State:**
        *   `const LIL_LISA_SERVER_URL = "{{ BASE_URL }}";` (Assuming Flask injects the env var into the template context, or retrieve differently).
        *   `let authToken = localStorage.getItem('authToken');`
        *   `let userInfo = { userIdHash: null, domain: null };`
    *   **API Helper Function:**
        *   Create `async function apiRequest(endpoint, method = 'GET', body = null, requireAuth = false)`:
            *   Constructs full URL: `${LIL_LISA_SERVER_URL}${endpoint}`.
            *   Sets headers: `'Content-Type': 'application/json'`.
            *   If `requireAuth` and `authToken` exists, add `'Authorization': \`Bearer ${authToken}\`` header.
            *   Handles `fetch` call with method, headers, and JSON.stringified body (if provided).
            *   Parses JSON response or handles errors (e.g., network error, non-2xx status).
            *   Returns response data or throws an error.
    *   **UI Update Functions:**
        *   Create `function updateUIForLoginState(isLoggedIn)`:
            *   Toggles `logged-in`/`logged-out` class on `<body>`.
            *   Updates `#user-info` text (e.g., show domain or greeting) if `isLoggedIn`, clears otherwise.
            *   Shows/hides relevant elements (`#login-btn`, `#logout-btn`, `#history-panel`).
    *   **Event Listeners:**
        *   Login Button (`#login-btn`): `onclick` shows the `#login-modal`.
        *   Login Modal Close: Hides `#login-modal`.
        *   Logout Button (`#logout-btn`): `onclick` calls a `logout()` function.

**Phase 2: Authentication Flow Logic (JS)**

1.  **Login Submit (`#login-submit-btn` listener):**
    *   Get email value from `#login-email`.
    *   Basic client-side validation (e.g., simple regex for format).
    *   Call `apiRequest('/api/auth/request_login', 'POST', { email: emailValue })`.
    *   On success: Display confirmation message in `#login-message`, maybe hide modal.
    *   On error: Display error message in `#login-message`.
2.  **Login Verification (Run on page load, specifically check for `/login_verify` path or param):**
    *   Check `window.location.search` for `?token=...`.
    *   If token param exists:
        *   Extract the `token` value.
        *   Call `apiRequest(\`/api/auth/verify_token?token=${token}\`, 'GET')`.
        *   On success (backend returns JWT, user info):
            *   `authToken = response.access_token;`
            *   `localStorage.setItem('authToken', authToken);`
            *   `userInfo = { userIdHash: response.user_id_hash, domain: response.domain };`
            *   `updateUIForLoginState(true);`
            *   **Call function to fetch history (Phase 4).**
            *   Redirect/clear URL parameters (`window.history.pushState({}, document.title, window.location.pathname);`).
        *   On error: Display persistent error message on page (e.g., \"Login link invalid or expired.\").
3.  **Session Verification (Run on every page load):**
    *   Create `async function checkSession()`:
        *   `authToken = localStorage.getItem('authToken');`
        *   If `authToken` exists:
            *   Try `apiRequest('/api/auth/verify_session', 'POST', null, true)`.
            *   On success:
                *   `userInfo = { userIdHash: response.user_id_hash, domain: response.domain };`
                *   `updateUIForLoginState(true);`
                *   **Call function to fetch history (Phase 4).**
            *   On error (e.g., 401 Unauthorized):
                *   Call `logout()` function (clears token, updates UI).
        *   Else (no token):
            *   `updateUIForLoginState(false);`
    *   Call `checkSession()` when the page loads (`window.onload` or DOMContentLoaded).
4.  **Logout Function:**
    *   Create `async function logout()`:
        *   Optionally call `apiRequest('/api/auth/logout', 'POST', null, true)` (ignore errors).
        *   `authToken = null;`
        *   `userInfo = { userIdHash: null, domain: null };`
        *   `localStorage.removeItem('authToken');`
        *   `updateUIForLoginState(false);`
        *   Clear history panel content.
5.  **Modify `sendQuery` Function:**
    *   Inside `sendQuery`, before the `fetch` call:
    *   Check if `authToken` exists.
    *   If yes, add the `Authorization: Bearer ${authToken}` header to the `fetch` options for the `/invoke` or `/invoke_stream` call to the backend.
    *   *Remove* the existing `/api/chat` Flask endpoint call logic.

**Phase 3: History UI & Core Logic**

1.  **HTML Structure (`main.py::index` return string):**
    *   Inside `#history-panel`: Add search input (`#history-search`), filter checkbox (`#history-filter-mine`), list container (`#history-list`).
2.  **CSS (`main.py::index` style block):**
    *   Style the history panel (layout, scrollbar, width, collapse button if desired).
    *   Style the search input.
    *   Style list items, group headers, active state, user indicator.
3.  **JavaScript Core (`main.py::index` script block):**
    *   **State Variables:**
        *   `let conversationHistory = [];`
        *   `let historyOffset = 0;`
        *   `let historyLimit = 25;` // Or desired page size
        *   `let isLoadingHistory = false;`
        *   `let hasMoreHistory = true;`
        *   `let currentHistoryFilter = 'domain';` // 'domain' or 'mine'
        *   `let currentHistorySearch = '';`
        *   `let historySearchTimeout = null;`
    *   **History Rendering Function:**
        *   Create `function renderHistory(items)`:
            *   Clears `#history-list` content.
            *   Groups items by relative date (Today, Yesterday, etc.) client-side based on `created_at`.
            *   Iterates through groups and items, creating HTML elements (group headers, item divs with title, user indicator based on `item.user_id === userInfo.userIdHash`).
            *   Appends elements to `#history-list`.
            *   Adds click listeners to items (call `handleHistoryItemClick`).
    *   **Append History Function:**
        *   Create `function appendHistory(items)`:
            *   Similar to `renderHistory` but *appends* new items to the existing list, potentially creating new date groups if needed.
    *   **Fetch History Function:**
        *   Create `async function fetchHistory(append = false)`:
            *   If `isLoadingHistory`, return.
            *   If `!hasMoreHistory` and `append`, return.
            *   `isLoadingHistory = true;`
            *   If `!append`: `historyOffset = 0; hasMoreHistory = true;` Clear `#history-list`.
            *   Construct API endpoint: `/api/history?offset=${historyOffset}&limit=${historyLimit}&filter=${currentHistoryFilter}`.
            *   If `currentHistorySearch`: add `&search=${encodeURIComponent(currentHistorySearch)}`.
            *   Call `apiRequest(endpoint, 'GET', null, true)`.
            *   On success:
                *   If `append`: `appendHistory(response);`
                *   Else: `renderHistory(response);`
                *   `historyOffset += response.length;`
                *   `hasMoreHistory = response.length === historyLimit;`
            *   On error: Display error message.
            *   `isLoadingHistory = false;`
    *   **Event Listeners:**
        *   Filter Toggle (`#history-filter-mine`): `onchange` updates `currentHistoryFilter`, calls `fetchHistory(false)`.
        *   Search Input (`#history-search`): `oninput` uses `clearTimeout(historySearchTimeout)` and `setTimeout` (debounce, e.g., 300ms) to update `currentHistorySearch` and call `fetchHistory(false)`.
        *   Scroll Listener (`#history-list` or panel): `onscroll` checks if near bottom (`scrollHeight - scrollTop <= clientHeight + threshold`) and if `!isLoadingHistory && hasMoreHistory`, then calls `fetchHistory(true)`.

**Phase 4: History Interaction Logic (JS)**

1.  **History Item Click Handler:**
    *   Create `function handleHistoryItemClick(event)`:
        *   Get the clicked item element.
        *   Retrieve the associated `first_query` data (store it as a data attribute on the element during rendering).
        *   Set `#query-input.value = first_query;`.
        *   Maybe visually highlight the selected item.

**Phase 5: Refinement & Testing**

1.  **Testing:** Test login flow (email sending, link clicking, errors), auto-login, logout, history loading (initial, scroll, filter, search), clicking history items, sending chat messages while logged in/out.
2.  **UI Polish:** Ensure smooth transitions, clear loading states, responsive design.
3.  **Error Handling:** Test backend API errors, network errors, invalid token scenarios.
