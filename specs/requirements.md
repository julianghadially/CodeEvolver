# CodeEvolver Agents Requirements

## Components
### Api
- /change_request 
    - change_request: text description of a change request
    - program_id: new program id
    - parent_program_id: program id of parent(s). Can have 1 or 2 parents. 
    - Kicks off auto-coder agent. Fetches program_json for the prompts themselves
- /connect-git
    - will include private but assume public for now.
### Program Db
    - client_id: Internal client id
    - program_id
    - parent_program_id
    - program_json: dspy optimized program json, representing all prompts for each module
    - Db centralizes prompt changes for direct editing by external optimizer. Preference for mongo db
### Auto-coder
- claude agents sdk
- Zero human in the loop. 
- Can make any edit (future: accepts constraints)
- Dedicated execution environment such as Daytona (or claude's own execution environment), so that each agent gets its own execution environment, so it can edit and run code from different git branches.
- Access to repository code for one single branch (while other auto-coder workers can access git code for a separate branch simultaneously).
### Run Program
- Runs updated program as a fixed DSPY Adapter
### Git branching bot
- creates a new branch for the program, program_id

### External
- codeevolver-gepa: package for orchestrating gepa with CodeEvolver capability. Likely to import into this repository