# Prompt-lesson catalog: appworld

128 lessons in 14 themes. By label: Task 5, Domain 25, Tool 31, Procedure 29, Decision 30, Output 8.

## Lessons

### T1: REPL execution and step trace

#### L1: Do not fabricate or imitate the step history
Label: Procedure
Description: Only the single code block of the current step is executed. Invented step headers, outputs, passwords, tokens or ids for past or future steps are discarded and are not real. Emit one fresh code block and re-fetch real data instead of reusing invented values.

#### L2: Reference only variables from executed successful steps
Label: Procedure
Description: The executed trace is the only record of what happened. Use only variables assigned in a step that actually succeeded. If a needed value is not visible, re-fetch or re-create it instead of assuming it exists.

#### L3: The API object is pre-imported
Label: Tool
Description: The API handle already exists in the REPL. Importing it is wrong and fails.

#### L4: Never rebind, shadow or delete the API handle
Label: Tool
Description: The API object is the only handle to the environment. Reusing its name as a variable, loop or except name, or deleting it, breaks every later step irrecoverably. Call methods on it directly.

#### L5: Never terminate the REPL
Label: Tool
Description: exit, quit, sys.exit or raising SystemExit crashes the step. The only way to finish is the task-completion API.

#### L6: Print results to see them
Label: Tool
Description: The REPL shows only what is printed. Wrap any call whose result is needed, especially discovery calls, in print, and do not call a discovery API repeatedly without printing it.

#### L7: Truncated output: compute in code
Label: Tool
Description: Long printed output is cut off in the step view, so items past the cut are invisible. The REPL variable holds the full data. Filter, count and loop over it in code and print only the result, rather than reading or counting from printed text.

#### L8: Persistent REPL state
Label: Tool
Description: Variables and stored data persist between code steps, so earlier results can be reused instead of re-fetched.

#### L9: Keep a progress notes record
Label: Procedure
Description: Maintain a small dictionary of completed subgoals, found ids and pending conditions so earlier facts are not lost across steps.

#### L10: Bound fan-out loops over detail APIs
Label: Tool
Description: The environment caps total API calls at roughly 1000. Do not call a per-item detail API in an unbounded loop. Prefer batched list APIs and loop only over the items actually needed.

#### L11: Split oversized code blocks
Label: Procedure
Description: A syntax error about an unclosed delimiter or unexpected end of input means the code was cut off. Redo the work in smaller steps with one action per step and all brackets and strings closed.

#### L12: Avoid wasted steps
Label: Decision
Description: The step budget is limited. Do not resubmit an identical call that already succeeded or already failed, and do not re-read docs or re-fetch data already held. Spend steps on new information, with at most one re-query to verify a change. After a failure, change the approach before retrying.

#### L13: Commit once evidence suffices; budget endgame
Label: Decision
Description: When the target is identified, perform the required actions rather than exploring for perfect confirmation. When few steps remain, complete the task if it is done. Otherwise do only the single most essential remaining action.

#### L14: Never retype values; reuse stored variables
Label: Procedure
Description: Extract ids, passwords, emails and target sets from responses in code and pass the variables on. Do not retype or hardcode them from truncated or printed output or from memory, including inside destructive loops.

### T2: Authentication and credentials

#### L15: Login username differs by app
Label: Domain
Description: Login uses the supervisor's email as username for every app except the phone app, which takes the phone number as a string. Using the email there returns 401.

#### L16: Password source and keying
Label: Domain
Description: Account passwords come from the supervisor's stored password listing. Each entry holds only the app name and the password, with no email.

#### L17: Copy passwords literally
Label: Procedure
Description: Copy the password exactly, including special characters such as brackets, braces, carets, ampersands and backslashes. Unmatched brackets are literal and must not be 'fixed'.

#### L18: Failed login: no sign-up or password reset
Label: Decision
Description: On a failed login, re-read the password and re-check the username field. Do not sign up, send reset or verification codes, or reset the password unless the task requires it, because these create unwanted account and email records.

#### L19: Capture the access token and pass it explicitly
Label: Tool
Description: login() returns a dict and does not store the token. Take its access_token field, keep it in a variable and pass it on every authenticated call. A 401 'token missing' error means it was omitted. Passing the whole dict or a truncated printed token also fails.

#### L20: Reuse tokens; re-login only after 401
Label: Decision
Description: A stored token stays valid across steps. Do not log into the same app again unless a call actually returned 401 or Unauthorized.

### T3: API discovery and usage

#### L21: Never guess API or parameter names
Label: Procedure
Description: Take exact API names from the API descriptions listing and exact parameter names from the API doc. Guessed names cause errors or are silently ignored and give wrong results.

#### L22: Pick from the descriptions list; read docs selectively
Label: Procedure
Description: The one-line API descriptions are enough to choose an API. Open the full doc only for the API about to be used. Never loop over the docs of all APIs.

#### L23: Long API listings get truncated
Label: Tool
Description: Verbose API description output is cut off in the trace, hiding later APIs. Print only the compact list of API names to see all of them.

#### L24: Docs are reached through the docs object
Label: Tool
Description: On a non-existent API error, rediscover the exact name through the API-descriptions call. Docs are accessed through the dedicated docs object, not on the individual app object.

#### L25: Known API naming traps
Label: Tool
Description: Specific names matter: the file app is file_system, notes are listed via search_notes, and directories and files via show_directory and show_file. move_file uses source and destination path parameters. Playlist and album songs come from a song_ids field, and API names are singular (e.g. remove_song_from_queue).

#### L26: Keyword-only API calls
Label: Tool
Description: All API calls take keyword arguments only.

#### L27: Inspect response shape before using fields
Label: Procedure
Description: Before extracting or filtering on any field, print the API's response schema or one sample's keys. Field names and types differ between endpoints (singular vs plural, song_id vs id, song_ids vs songs). A guessed field name silently yields empty results. If a union of sources equals only one source's count, a field name was probably wrong.

#### L28: Recover from errors using the message and docs
Label: Procedure
Description: Read the traceback and error message, which usually point to wrong API names, parameter names or types. Re-list API names or re-read the doc before retrying.

#### L29: Look up real ids; never use display values as ids
Label: Procedure
Description: Use show or list APIs to find the actual id field. Do not use display values such as card numbers or emails as ids.

#### L30: Use the purpose-specific native API for the task verb
Label: Decision
Description: Use the app's built-in API that matches the intent (download song, create playlist, like, add friend, send text, send reminder). Do not substitute a generic create, update or file-write call for a different intent.

#### L31: A returned message field means success
Label: Tool
Description: The runtime raises an exception on real failures. A response that returns normally with a message field is a success, so its presence is not a failure check.

#### L32: Retry with other parameter values before switching APIs
Label: Decision
Description: When a call fails because of a parameter issue (wrong id, insufficient funds, expired card), try other valid parameter values before switching to a different API. Indirect approaches create extra records.

### T4: Pagination, search and retrieval behaviour

#### L33: Paginate every list and search API until a short page
Label: Tool
Description: List, search and library APIs return only about 5 items per page by default and drop the rest. Request the maximum page size and loop the page index until a page is short or empty before counting, filtering or acting. A result of exactly the page size means more exist.

#### L34: Confirm the pagination parameter name and the real page-size cap
Label: Tool
Description: Read the doc for the exact page parameter name (commonly 0-indexed page_index) and the maximum page_limit (often 20). The server silently caps larger values, so a short page then does not prove exhaustion. If every page returns the same rows, the parameter name is wrong.

#### L35: Search is fuzzy; require exact identity match
Label: Tool
Description: User, contact and playlist search return partial or fuzzy matches, so a non-empty result proves nothing. Filter to exact equality on a unique field (email, phone, id) before concluding something exists, and act on that exact set.

#### L36: Search query ranks but does not filter
Label: Tool
Description: A search or query parameter orders results by relevance and returns everything else that passes the other filters. Re-check each returned record's actual fields (e.g. description keyword, owner, category) and keep only real matches.

#### L37: Artist-filtered song search is unreliable
Label: Tool
Description: Searching songs by artist is not an exact filter. It can return other artists' songs and omit the target's. Paginate the unfiltered catalog and keep songs whose artist list contains the wanted artist ids.

#### L38: Derive answers and condition checks in code
Label: Procedure
Description: Compute answers (max, sum, counts) and per-candidate yes/no conditions programmatically with exact field matches. Do not eyeball long or loosely matched output.

#### L39: Empty or zero results signal a bug
Label: Decision
Description: If a category the task implies is non-empty comes back empty or zero, do not submit it. Likely causes are a wrong date or year, a too-narrow filter, a wrong field name or API, or unpaginated data. Print a raw sample, fix the cause and re-investigate.

#### L40: Find the record that states the value
Label: Procedure
Description: When a task refers to a value in a message, note or email, retrieve the specific record that explicitly states it. Sort by date, widen the window or paginate if needed, and do not take the first number seen.

### T5: Dates and time windows

#### L41: Use the simulated current date, computed in code
Label: Tool
Description: The environment has its own clock, separate from real time. Read the current date from the phone app's date/time API and compute relative phrases (today, this week, so far, last month, this year) from it in code. Never use the system clock or assume or hardcode the year.

#### L42: Current-date response format
Label: Tool
Description: The current-date call returns a dict with a date string like 'Weekday, Month DD, YYYY' and a time string. It has no year or datetime key, so parse the year from the date string.

#### L43: Last N days including today is N-1 days back
Label: Domain
Description: 'Last N days including today' spans N days in total, so the offset is N-1 days, not N.

#### L44: Window cutoff keeps the time component
Label: Decision
Description: When computing a lower bound for a day window, keep the time-of-day component of the current moment rather than flooring to midnight.

#### L45: Window cutoff at start of the boundary day
Label: Decision
Description: For a day-window lower bound, use midnight at the start of the boundary day. Subtracting whole days from the current moment keeps the time of day and can drop early records on the first day.

#### L46: Upper bound is the end of today
Label: Domain
Description: When filtering up to today, set the upper bound to the end of the day or the actual current time, not midnight, or today's entries are excluded.

#### L47: Match the year as well as the month or unit
Label: Decision
Description: For a relative period such as 'February and March of this year', check the year against the current date as well as the month.

#### L48: Time-of-day and occasion qualifiers restrict the match
Label: Decision
Description: Words like 'yesterday evening' or 'at lunch' restrict to that part of the day (evening is roughly 17:00-23:59) and to the record's description or occasion. Several events can share a date, so apply the hour range and description together.

### T6: Building the exact target set

#### L49: Every stated qualifier is a filter; intersect them
Label: Decision
Description: List every qualifier in the instruction (relationship, time or position, category, state, owner, requester clause, 'only', 'except', 'not'). Check each per candidate and build the target as the intersection of all of them. Never act on a whole container or drop a condition.

#### L50: Do not invent unstated filters
Label: Decision
Description: Apply only the conditions the task states and act on every matching item. Do not exclude items using an aggregate or other field the task never mentioned.

#### L51: Make exactly the required changes
Label: Task
Description: Perform exactly the state changes the task requires and no others. Extra changes fail a task as surely as skipped ones.

#### L52: Disambiguate to one interpretation or record
Label: Decision
Description: When several candidates could match or the wording is ambiguous, verify the condition explicitly and act on the single correct record or interpretation, not on all candidates to be safe.

#### L53: Enforce qualifiers as explicit code filters
Label: Procedure
Description: Re-assert every qualifier as an explicit filter on fetched results instead of relying on an API parameter. A wrong or off-by-one parameter silently expands the target set.

#### L54: Compare sets by id membership, not counts
Label: Procedure
Description: Compute keep or remove sets by id membership. Equal list lengths do not mean equal sets.

#### L55: Print the full target set and count before bulk operations
Label: Procedure
Description: Gather the complete candidate set, apply all filters, and print every id with its count, untruncated, before acting. Then use the stored variable or recompute it in the action step.

#### L56: Parse per-record status attributes and filter on them
Label: Procedure
Description: When records carry a boolean or status attribute and the task asks for those with a given value, parse every record and keep only matching ones. Do not take the first or assume all match, and print the filtered set and its count.

#### L57: Replace or reset with a minimal diff
Label: Procedure
Description: To make a collection equal a target set, add only the missing items and remove only the extras. Leave common items untouched. Do not clear and rebuild, which creates spurious change records. Verify the result equals the target.

#### L58: Apply the criterion in the requester's message
Label: Procedure
Description: For 'as per X's request' or 'reply to X', first find and read X's relevant (most recent) message, extract the specific filter it names, and act only on matching items rather than the whole list or note.

#### L59: Prefer existing items over creating new ones
Label: Decision
Description: When asked to play, share, use, find or open an item meeting a criterion, pick an existing one, judging candidates by actual properties. Create a new one only if the task explicitly asks. If a search finds nothing, re-check the criteria and all candidates, including small ones, before concluding.

#### L60: Cumulative 'so far' sets and progress phrases
Label: Decision
Description: 'Played so far', 'up to now' or 'including the current one' means all items up to the current position or all with played or completed status, not just the current or last item and not the whole list. Find the position from the current-state API.

#### L61: Complete every clause of multi-action tasks
Label: Decision
Description: List every distinct action or answer in the instruction up front. Before finishing, confirm each is done, not just the first.

### T7: Contacts, relationships and cross-app identity

#### L62: Relationship nouns mean phone contacts, not the payment app's friends
Label: Domain
Description: 'My friends', 'coworkers', 'roommates', 'family' and similar words denote phone contacts filtered by relationship, fetched completely with pagination, even without 'in my phone'. The payment or other app's friend list is a different set. Use it only when the task is about managing or checking that list.

#### L63: Contacts have a plural relationships list field
Label: Domain
Description: Phone contacts carry a list-valued relationships field (friend, coworker, parent, roommate and so on), with no singular field. Filter with list membership. A singular-field check returns nothing and yields a bogus empty set. A specific relationship excludes contacts with other relationships.

#### L64: Match people across apps on any field
Label: Domain
Description: The same person may have different emails, phones or name spellings in different apps. Match a contact to an account in another app by searching its directory and accepting a match on name, email or phone, not email alone.

#### L65: Group or trip relationship word is a filter
Label: Decision
Description: When recording trips or groups 'with my coworkers/friends/family', include only those whose members all belong to that relationship set. Skip any that include anyone else.

### T8: Spotify collections and music data

#### L66: Distinct API names mean distinct collections
Label: Domain
Description: APIs differing by a word return different sets: song library vs liked songs, album library vs liked albums, sent vs received, app friend graph vs phone contacts. Match the task's exact noun and do not treat them as synonyms.

#### L67: 'Liked songs' means only individually liked songs
Label: Domain
Description: 'My liked songs' is exactly what the liked-songs API returns. Liked-albums and liked-playlists return the albums and playlists, not the songs inside them. Do not union in songs of liked albums, playlists or libraries.

#### L68: A 'liked' qualifier uses the subject's own entity type
Label: Domain
Description: For 'songs from collection X that I have liked', take the liked-songs list filtered to X. For other entity types, use the matching liked collection (liked albums for albums). Never substitute another entity's collection or all of X's songs.

#### L69: 'From my libraries' is scope or provenance, not a union
Label: Domain
Description: In phrases like 'liked songs from my song/album/playlist libraries', the subject is the noun's set. The library wording only describes or restricts the origin and never adds all items of each library.

#### L70: like_count is a global total
Label: Domain
Description: like_count counts likes from all users and does not show whether the supervisor already liked the item. Liking an already-liked item does not error, so every qualifying item should be liked.

#### L71: Song list records lack a genre field
Label: Domain
Description: Songs from the liked-songs and library APIs have no genre. Get it from the full song record or the album record, where it is a single string. There is no genres list.

#### L72: 'Enough' playlist means total duration
Label: Domain
Description: A playlist is enough for an activity when the summed duration of its songs meets the required time, taken from the task or a referenced note. A single long song can suffice, so item count is irrelevant. For downloads, count only the downloaded subset.

#### L73: Plan minutes vs song seconds
Label: Domain
Description: Plan durations are in minutes while song duration fields are in seconds. Convert the song side, not the plan, which would inflate the target and make every playlist look too short.

### T9: Venmo payments

#### L74: Transaction direction comes from sender and receiver
Label: Domain
Description: Venmo amounts are always positive. 'Paid/sent/spent' means transactions where you are the sender, and 'received' means where you are the receiver, combined with the other qualifiers.

#### L75: Sender and receiver are objects
Label: Tool
Description: Sender and receiver in transactions are objects with an email field, not strings. Compare their email to your own.

#### L76: Send money directly with the transaction API; avoid balance top-up
Label: Tool
Description: To pay, call the transaction-creation API directly, with the balance by default or a payment card id. add_to_venmo_balance creates bank-transfer, email and receipt-file side effects that count as unintended changes, so do not pre-fund the balance or top up as a workaround for a failed payment. Approving payment requests also accepts a card id directly.

#### L77: Explicit refill request means use the balance top-up call
Label: Decision
Description: If the task asks to refill or add to the Venmo balance, even as a conditional suggestion, use the top-up call with a non-expired card that has sufficient funds.

#### L78: Top-up only as a last resort when all funds fail
Label: Decision
Description: If the balance and every non-expired card are insufficient and the task does not forbid it, balance top-up is acceptable as a final fallback. Try direct payment with every card first.

#### L79: Use the proper funding API when funding is authorized
Label: Tool
Description: Discover the correct funding API. A bank transfer from a linked account is distinct from a card-based balance top-up, so one card failing does not mean funding is impossible.

#### L80: List cards and pass the chosen card id
Label: Tool
Description: List payment cards with the card-listing call and pass the chosen card's id as the payment-card parameter of the transaction call. Using a card is not a top-up.

#### L81: Fall back across non-expired cards
Label: Decision
Description: If the balance or one card cannot cover a payment, retry with each other non-expired card in turn. A single card may not cover every payment in a batch. Do not skip the payment, switch apps or top up.

#### L82: Card validity runs through the end of the expiry month
Label: Domain
Description: A card is valid through the end of its expiry month and year, checked against the simulated current date.

#### L83: Failed payment attempts create no records
Label: Tool
Description: A payment that errors for insufficient funds or an expired card creates no records, so it is safe to try the next card.

#### L84: Link an unlinked funded supervisor card when all cards fail
Label: Procedure
Description: If every card on the payment app is expired or lacks funds, look in the supervisor's wallets across apps for a non-expired funded card. Add it with its exact field values, never an invented one, and use it for the payment. Do not complete the task unfunded.

#### L85: Payments do not require friendship
Label: Tool
Description: Payment requests and transactions do not require the recipient to be a friend, and a null friends-since value in user search is normal. Adding a friend creates an unintended friendship record on payment tasks.

#### L86: Sending is not requesting
Label: Domain
Description: 'Send money to X' means creating a transaction that pays them. A payment request asks them to pay you, which is the opposite direction and never a substitute.

#### L87: Splits among 'X and me' include the user
Label: Decision
Description: A split 'among X and me' or 'between us' counts the user, so divide by the number of others plus one, even though requests go only to the others.

#### L88: Judge 'already paid' by exact share and description
Label: Decision
Description: To decide whether someone already paid their share of an event, look for a received transaction whose amount equals their exact share and whose description matches the event. Any unrelated earlier payment is not proof.

#### L89: Event totals include all payers and the user's own payment
Label: Decision
Description: When summing payments for an event, select by description, date and payee. Include all senders, even those not in the phone contacts, and the user's own payment (which may be visible only in the payee's sent transactions). Paginate fully.

### T10: Splitwise expenses

#### L90: Owed-to direction: the person owed is the payer
Label: Domain
Description: The payer is whoever paid and the debtors are whoever owes. 'I owe X' makes X the payer and you the sole debtor, and 'X owes me' is the reverse. Do not default to making yourself the payer.

#### L91: One non-grouped expense per owed person
Label: Domain
Description: To record money the user owes, create one separate expense per owed person. For an individual or non-grouped expense, omit the group id.

#### L92: Equal split lists include the payer as a debtor
Label: Domain
Description: In 'owed equally by [A, B, C]', the listed people are the full debtor set including the payer, whose own share nets to zero. Pass all of them as debtors and omit explicit debt amounts unless the note states unequal amounts.

#### L93: Payer is excluded from the debtors; shares sum to the total
Label: Domain
Description: In a shared-expense record, the payer must not also be a debtor, and the debtor shares must sum exactly to the total (equal unless told otherwise).

#### L94: Skip groups with no stated owed amount
Label: Decision
Description: Record an expense only for a group or trip whose source note explicitly states an owed amount. Skip the others instead of creating zero or placeholder expenses.

#### L95: Group expense listing shows only the user's expenses
Label: Tool
Description: show_group_expenses returns only expenses where the authenticated user is payer or debtor, so an empty result does not mean nothing was recorded. Verify a recorded expense with show_expense using the returned id, without re-recording.

#### L96: Splitwise deletion is a soft delete and still counts
Label: Tool
Description: A deleted expense still counts as a change, so delete-and-re-record doubles the records. Verify inputs before recording, and prefer updating over deleting.

### T11: Files, notes and exports

#### L97: Use native move/copy with date retention
Label: Tool
Description: Move files with the file system's native move call and retain_dates=True, and likewise for copy. Otherwise timestamps reset to the current time. Do not emulate a move with create plus delete, which loses the original identity and dates.

#### L98: Classify files by original created_at, checking the full condition
Label: Procedure
Description: Record each file's original created_at before any move or rename, since moving can reset it, and classify by that value. Check the whole date condition, e.g. 'this year' needs the year as well as the month.

#### L99: Strip trailing slashes when deriving path names
Label: Tool
Description: Strip trailing slashes before taking the last path component, since a trailing slash gives an empty name that corrupts derived paths. Assert the name is non-empty before use.

#### L100: Sanity-check destructive file targets
Label: Decision
Description: Before destructive file or directory operations, print the target paths and check each has a sensible non-empty name. Stop and re-list if the list is empty, includes the parent directory or is at the wrong level.

#### L101: Preserve literal tilde paths
Label: Output
Description: When a task gives a path starting with ~, pass it literally to file APIs. Expanding it to a home directory breaks the match with the stored tilde path.

#### L102: Unreadable binary files
Label: Domain
Description: A file whose content is a binary placeholder (e.g. starts with 'binary:') is unreadable. Do not copy, rename, decompress or create directories to read it, and do not hunt for its content elsewhere, guess it or copy it from sibling files. Aggregate only values actually read and answer from the readable files.

#### L103: Count checkbox markers by leading position
Label: Procedure
Description: To count checked or unchecked items in a note, count only lines that begin with the marker (after optional whitespace or bullet), once per line. Title or legend lines may embed the markers and inflate substring counts. Print the matched lines to verify.

#### L104: CSV fields written without quote characters
Label: Output
Description: Write CSV rows and headers as plain comma-separated text without surrounding double quotes. Build the lines manually. The grader splits naively on commas and does not strip quotes, so stray quote characters end up in the values.

#### L105: CSV headers and separators exactly as specified
Label: Output
Description: Use the exact header names and separators the task specifies (e.g. comma plus space), using a separator only inside the field it describes. Quotes around names in the task text are emphasis, not quoting instructions.

#### L106: Ordered log notes: follow the ordering convention
Label: Domain
Description: Monthly or ordered log notes are normally newest-first, so the new entry goes directly after the header line and before existing entries. Check which order the note uses, preserve existing entries, and re-read afterwards to verify.

#### L107: Match the existing entry format when adding entries
Label: Output
Description: A new entry must copy the existing entries' exact format (markers, separators, units, spacing), keep the title or header as the first line, and add no blank lines between entries.

#### L108: Rewrite ordered notes in full; avoid the append/prepend API
Label: Tool
Description: The note append-or-prepend API inserts at the very start or end of the content, so prepend lands before the header and append after all entries, and it adds a blank-line separator. Rewrite the whole content with the note-update API, passing the original title, tags and pinned status explicitly.

#### L109: Copy every attribute when cloning or recreating a record
Label: Procedure
Description: For 'same as X except Y', recreating or correcting records, or creating a record meant to resemble existing ones, print the original and copy all attributes (title, content, tags, pinned, receiver, description, privacy flag and so on), changing only the stated difference. Pick the originals by applying every qualifier and recreate one per original.

#### L110: Move by copy: omit the positional order field
Label: Domain
Description: For apps without a move API, a move is a copy of every content attribute into the destination followed by deleting the original. Do not copy the order or position index, which shifts the destination's existing records. Append at the end instead.

#### L111: Order of a copy-based move
Label: Procedure
Description: Compute the target set once, create all copies, verify them read-only, and only then delete the originals. Never create records in the source collection or recreate a deleted original. Fix a mistake afterwards only by removing a wrong destination copy.

#### L112: Share the exact returned link, once, for the final record
Label: Output
Description: When sending a record's link, copy the link field value the API returns verbatim and never construct a URL from the id. Send it once, only after the record is verified as final.

### T12: Side effects and state integrity

#### L113: Touch only the apps and records the task requires
Label: Decision
Description: Do not send emails, messages or notifications, create files, transfer money, add friends or modify unrelated records unless the task requires it. Any unintended change counts against the result. If an API call fails, retry within the relevant apps rather than using unrelated ones.

#### L114: Information tasks stay read-only
Label: Task
Description: Information-seeking tasks must make no state changes. Use only read-only show, search, get and exists APIs and compute the answer in code. Do not decompress archives or create directories to read a file.

#### L115: Never re-run a successful mutating call
Label: Procedure
Description: Do not repeat a send, create, update or delete that already succeeded, since it creates duplicate records. Reuse returned ids. If the output was cut off or empty, confirm the state with a read-only call instead of repeating the mutation.

#### L116: After a partial failure, re-query and handle only what remains
Label: Procedure
Description: If a step errors partway, some actions may already have executed, and an 'already exists' error means an earlier step partly completed. Re-read the state and skip items already done. Treat an item as done only if all its identifying fields match (receiver, amount, description, privacy), not recipient alone. Do not rerun the whole batch.

#### L117: Do not swallow errors in batch actions
Label: Procedure
Description: Do not wrap each call of a batch in a try/except that silently continues, because swallowed errors look like success. After the batch, re-read the state in a separate step, confirm every intended item changed, and fix or retry any that did not.

#### L118: Delete temporary artifacts created during recovery
Label: Decision
Description: Delete any temporary files or directories created while recovering from an error before completing, since leftovers count as unintended changes.

### T13: Verification and completion

#### L119: Verify the end state by re-querying
Label: Procedure
Description: Before finishing a state-changing task, re-read the affected records read-only and confirm the final state matches the request exactly. Check values, sort order, format and the complete set. Do not re-run the action to verify.

#### L120: Final check for extras and side effects
Label: Decision
Description: Before completing, confirm every condition was applied, the affected set has no extras or omissions, and no unintended side effects exist (emails, files, transfers, balance additions, records). Undo extra unintended changes so the end state matches exactly.

#### L121: Do not complete after errors, before real work or on failed verification
Label: Decision
Description: Do not call task completion when the previous step errored, when only documentation lookups were done, or when verification shows the goal is not met (empty results, missing items, leftover records, mismatched counts). Fix and verify first.

#### L122: Completion is terminal and goes alone in the last step
Label: Procedure
Description: Completing the task ends and scores it immediately. Call it only when everything is done and verified, as the sole action of its own code block. Never plan 'do X then complete' in one block.

### T14: Task type and answer format

#### L123: Action tasks: perform the action via the API and submit no answer
Label: Task
Description: Imperative tasks (send, reply, create, update, delete, download, play, advance) require actually doing the action through the matching API. Then call completion with no answer argument. Passing an answer fails a pure action task.

#### L124: Hybrid tasks: the value goes inside the action
Label: Task
Description: If the task asks to send or record something that contains a value ('reply with X', 'text them the name'), put the value in the message or record and submit no answer. A question word there describes the content, not a return value.

#### L125: Hybrid tasks: act, then also return the value
Label: Task
Description: A task that both performs an action and asks for a value still needs the value passed as the answer to completion.

#### L126: Information answers are the bare value only
Label: Output
Description: For information tasks, pass only the requested value (a number, name, date or short comma-separated list) as the answer, not a sentence, label, confirmation, list marker or numbering.

#### L127: Numeric answers: bare raw number
Label: Output
Description: Pass one bare numeric value with no currency symbol, units, labels or forced two-decimal formatting, because grading is by exact match. Do not pass several values.

#### L128: Clean extracted answers; quotes verbatim
Label: Output
Description: When the answer comes from note, file or message content, strip bullets, headers, numbering and attribution lines in code so only the bare value remains. Keep quoted text exactly verbatim, since a paraphrase fails equality checks.

## Notes from discovery (ignored by the parser)

- Two lessons about 'last N days' window cutoffs are kept separate because they prescribe opposite behaviour: keep the time-of-day component of the current moment versus use midnight at the start of the boundary day.
- Two Splitwise lessons conflict and are kept separate as opposite variants: list the payer among the debtors in an equal split versus exclude the payer from the debtors.
- Two hybrid-task lessons conflict and are kept separate: put the value in the action and submit no answer versus act and also pass the value as the answer.
- Venmo top-up has three variants kept separate: avoid it as a workaround, use it when the task explicitly mentions a refill, and use it only as a last resort when every card and the balance fail.
- Merged lessons: all repeated or identical-step avoidance, repeated docs reading and the step-budget idea are in one 'wasted steps' lesson, while duplicate mutations and partial-failure handling stay separate because they concern record duplication.
- Merged lessons: retyping values from truncated or printed output, extracting credentials in code and reusing variables in destructive loops are one lesson. Printing the target set before acting stays separate as a verification step.
- The lessons on whether to cut the window at midnight and on which collection counts as 'liked' reflect small differences in wording between the sources. A reviewer may want to check the Spotify 'from my libraries' lesson, since one phrasing treats it as an intersection scope and another as pure provenance.
- (validator) possible provenance wording 'later' in: The API object is the only handle to the environment. Reusing its name as a variable, loop or except name, or deleting i
- (validator) possible provenance wording 'earlier' in: Variables and stored data persist between code steps, so earlier results can be reused instead of re-fetched.
- (validator) possible provenance wording 'earlier' in: Maintain a small dictionary of completed subgoals, found ids and pending conditions so earlier facts are not lost across
- (validator) possible provenance wording 'later' in: Verbose API description output is cut off in the trace, hiding later APIs. Print only the compact list of API names to s
- (validator) possible provenance wording 'The run' in: The runtime raises an exception on real failures. A response that returns normally with a message field is a success, so
- (validator) possible provenance wording 'earlier' in: To decide whether someone already paid their share of an event, look for a received transaction whose amount equals thei
- (validator) possible provenance wording 'earlier' in: If a step errors partway, some actions may already have executed, and an 'already exists' error means an earlier step pa
