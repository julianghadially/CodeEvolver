# Prompt-lesson catalog: alfworld

57 lessons in 7 themes. By label: Task 3, Domain 12, Tool 8, Procedure 8, Decision 25, Output 1.

## Lessons

### T1: Action interface and feedback semantics

#### L1: Action vocabulary and preconditions
Label: Tool
Description: The environment accepts a fixed set of verbs: go to, take from, put, open, close, use, heat/cool/clean with an appliance, examine, inventory and look. Each has a precondition. Take, put, open and close need the agent at the receptacle, and heat/cool/clean apply to a held object with the matching appliance.

#### L2: Never use 'move' for placement
Label: Tool
Description: To place a held object, use the listed 'put <obj> in/on <recep>' form. The 'move <obj> to <recep>' verb never works and must not be chosen.

#### L3: 'Move' form is the placement verb
Label: Tool
Description: The admissible list offers a 'move <obj> to <recep>' form for placement. It is translated into a working placement before execution, so choose it when placing a held object.

#### L4: Move as fallback placement verb
Label: Tool
Description: If both 'put X in Y' and 'put X on Y' return 'Nothing happens', try the alternative placement verb 'move X to Y'.

#### L5: Preposition depends on receptacle type
Label: Tool
Description: Use 'put X in' for containers (fridge, cabinet, drawer, microwave, safe). Use 'put X on' for surfaces (shelf, countertop, desk, dresser, bed, stoveburner, sinkbasin, diningtable).

#### L6: 'Nothing happens' means no effect; do not repeat, vary
Label: Decision
Description: A 'Nothing happens' response means the action had no effect, usually an unmet precondition (wrong location, object not held, step not done, container closed, wrong object type). Do not repeat it unchanged. Change the location, preposition, instance or an earlier step.

#### L7: Silent success of state-change actions
Label: Tool
Description: Heat, cool and clean can succeed while the feedback is terse or says 'Nothing happens'. Check progress (goal progress, or examining the object) before concluding failure or repeating the action. If the object is already hot/cool/clean, move on.

#### L8: Verify processing by examining the object
Label: Procedure
Description: Before placing a processed object, examine it. Hot, cool or clean confirms success. Normal or dirty means the process failed and must be re-applied at the right appliance.

#### L9: Missing 'go to' means already there
Label: Domain
Description: If 'go to X' is not in the admissible list, the agent is already at X or cannot reach it. Interact with X directly (place, examine, open) instead of navigating again.

#### L10: Retry failed navigation
Label: Procedure
Description: If 'go to' returns 'Nothing happens', the agent did not arrive. Retry until the observation confirms arrival before acting at that location.

#### L11: Trust text over camera image
Label: Domain
Description: The camera image may show a different room, lag, or contradict the text. Treat the text feedback, admissible actions, action history and tracked state as ground truth, and do not spend reasoning reconciling the discrepancy.

#### L12: Do not deliberate over a missing desired action
Label: Decision
Description: If a desirable action is not in the admissible list, pick the best available action instead of reasoning about why it is missing.

#### L13: State memory legend
Label: Tool
Description: The state memory shows what is held, the current location, searched receptacles with their contents, closed or unopened containers, unvisited receptacles and no-effect actions. An empty marker means the receptacle was opened and had nothing.

#### L14: End every response with the action object
Label: Output
Description: Every response must end with the structured (JSON) action, never reasoning alone, because the action is extracted from it.

### T2: Holding and object identity

#### L15: One object held at a time
Label: Domain
Description: The agent can hold only one object. If 'take' is missing from the admissible actions, the hands are full, so place the held item first.

#### L16: Put down a wrongly held object
Label: Decision
Description: If the held object is the wrong type or was taken by mistake, it can never complete the task. Put it down (back where found) before taking another, and do not take it again.

#### L17: Exact object type match
Label: Decision
Description: Take only an object whose type exactly matches the task's object name. Look-alikes are different objects (cloth vs handtowel, mug vs cup, pot vs pan, plate vs bowl, box vs tissuebox, soapbar vs soapbottle). Keep searching if the exact type is not found.

#### L18: Take only the target and take it when seen
Label: Decision
Description: Do not pick up distractor objects unrelated to the task. Take the target as soon as it is seen.

#### L19: Go straight to a seen target
Label: Decision
Description: If the target was already seen at a location, go there and take it instead of searching elsewhere.

#### L20: Must hold object before placing
Label: Domain
Description: An object must be held before it can be placed or processed. Check the holding state.

#### L21: Do not take an object already at the destination (plain pick-and-place)
Label: Decision
Description: In plain pick-and-place tasks, an object already at the destination is not the one to take. Find another instance elsewhere.

#### L22: Target already at destination in state-change tasks
Label: Decision
Description: In heat/cool/clean tasks the target may already be at the destination. If so, take it out, process it, then put it back.

### T3: Heat, cool and clean processing

#### L23: Adjective means required process
Label: Task
Description: A task phrased 'put a hot/cool/clean X in Y' means: take the plain X, apply that process, then place it in Y. The adjective is not part of any object's name.

#### L24: Process while holding, at the appliance; not by putting inside
Label: Procedure
Description: To heat, cool or clean, keep holding the object and run the process action at the appliance (heat with microwave, cool with fridge, clean with sinkbasin). Putting the object into the appliance, or opening and closing it, does not process it.

#### L25: Must stand at the matching appliance
Label: Domain
Description: Cleaning needs a sink, heating a microwave and cooling a fridge, and the agent must first go to and stand at that numbered appliance. From elsewhere the action does nothing.

#### L26: Appliance must be closed to heat or cool
Label: Domain
Description: Heating and cooling only work while the microwave or fridge is closed, and the action opens it itself. Do not open it first. If it was opened while searching, close it before processing.

#### L27: Cleaning only at a sinkbasin
Label: Domain
Description: Cleaning needs a sinkbasin. A bathtub or bathtubbasin is a different receptacle and does not clean.

#### L28: Process failed at one appliance: try another instance
Label: Decision
Description: A process can fail at one appliance instance yet work at another (for example a different sink or sinkbasin). Try another instance while holding the object, and do not return to the failed one.

#### L29: Do not reprocess a confirmed object; go place it
Label: Decision
Description: Once the object is confirmed hot, cold or clean, go to the destination and place it. Do not process or examine it again.

#### L30: Unprocessed object may cause placement failure
Label: Decision
Description: If placing keeps failing in a heat/cool/clean task, the required process may not have been done. Do the process first, then place.

#### L31: Stop retrying a broken appliance or lamp
Label: Decision
Description: If processing or lamp use fails everywhere (examine still shows 'normal', lamp use has no effect), treat the device as broken. Stop retrying and place the object at the destination for partial credit.

#### L32: Do not open and close an appliance repeatedly
Label: Decision
Description: Open an appliance once to look inside, then close it once when done. Do not cycle the door.

#### L33: Object placed unprocessed cannot be retrieved
Label: Decision
Description: If the object was placed in the goal receptacle without being confirmed processed, it cannot be taken back out. Leave it for partial credit and stop retrying.

### T4: Look-at-object-under-lamp tasks

#### L34: Task requires using the lamp while holding the object
Label: Task
Description: For 'look at X under/with lamp' tasks, take X, go to the lamp (desklamp or floorlamp) and use it while holding X. Merely examining X, or putting it back, does not complete the task, so do not loop on examine.

#### L35: Keep the same object in hand
Label: Procedure
Description: Keep holding the target object until the lamp is on. Do not put it down or swap it for another instance. If it was put down, fetch it again.

#### L36: Go to the lamp's receptacle before using it
Label: Procedure
Description: Using the lamp only works at its location. If use did nothing because the agent is elsewhere, find the receptacle where the lamp is, go there, then use it.

#### L37: Lamp may be hidden in a closed container
Label: Domain
Description: The lamp is often inside a closed drawer or cabinet. When holding the target but no lamp is found, open closed receptacles on arrival to look for it.

#### L38: Lamp use fails at a spot: try other surfaces or lamps
Label: Decision
Description: A lamp may be usable from only one of the receptacles where it appears. If use fails at one, try the other places it appears or another lamp instance, and do not return to a failed spot.

#### L39: Lamp use may be transiently ineffective
Label: Tool
Description: A lamp 'use' with no effect is not necessarily permanent. The lamp may already be on or have failed transiently, so retry, and examine the object there anyway.

### T5: Search and exploration

#### L40: Systematic search without revisiting
Label: Procedure
Description: Visit unvisited receptacles systematically, keep track of what each held, and do not revisit or re-open receptacles whose contents are known. The step budget is limited, so spend it on unchecked places.

#### L41: Check open surfaces before closed containers
Label: Decision
Description: Check receptacles that need no opening first (countertops, tables, shelves, sidetables, stove burners, sink, dresser). Then open closed cabinets and drawers one by one.

#### L42: Open closed receptacles to reveal contents
Label: Decision
Description: Closed receptacles hide their contents. Open each unopened one on arrival, before leaving it, because the target or a needed device may be inside.

#### L43: Closed appliances can hide the target
Label: Domain
Description: The target may be inside a closed fridge, microwave or cabinet, and these contents are not in the scene overview. Check unchecked appliances before drilling through more cabinets or drawers.

#### L44: Exhaust unsearched receptacles before giving up
Label: Decision
Description: When the object is not found, even with steps running low, check every unsearched reachable receptacle, including visited-but-unopened containers, before giving up.

#### L45: All receptacles checked means target is absent
Label: Decision
Description: If every receptacle has been checked and the target was not found, it is probably absent. Stop searching and revisiting.

#### L46: Break loops and oscillation
Label: Decision
Description: If recent actions repeat without progress (moving back and forth, picking up and putting down the same object, repeating an action with an identical result), stop. Do something different: visit an unvisited receptacle, open a closed one, move to the next subgoal, or do the action that completes the task.

### T6: Placement and destination handling

#### L47: Task pipeline order
Label: Procedure
Description: Do the steps in order: find the target, take it, apply any required heat/cool/clean change, then go to the destination (opening it if needed) and place it. The transformation comes before the final placement.

#### L48: Placing requires standing at the receptacle
Label: Domain
Description: A put only works when the agent is at the destination receptacle. If a put did nothing, go to that receptacle and retry.

#### L49: Open a closed container before placing
Label: Domain
Description: Placement into a closed container fails. Open the container (drawer, fridge, cabinet) first, then place.

#### L50: Any instance of the destination type works
Label: Task
Description: The target is a type of object and receptacle. Any numbered instance of the receptacle type counts, while the object must be of the exact target type.

#### L51: Rejected placement: try another instance
Label: Decision
Description: If placement at a receptacle does nothing, that instance may not accept the object. Go to a different instance of the same receptacle type and place it there.

#### L52: Put succeeded but task not complete: wrong destination instance
Label: Domain
Description: A put that succeeds without completing the task means the destination instance is wrong. Re-read the task's target receptacle, take the object back if needed, and place it at a different instance.

#### L53: Persistent rejection at the only or correct target: stop
Label: Decision
Description: If the object and target receptacle are correct (or the only instance of the type) and the put keeps being rejected, do not redo the process or hunt for another receptacle. The simulator rejects it and the task may be uncompletable.

### T7: Two-object tasks

#### L54: Deliver objects one at a time
Label: Procedure
Description: For tasks needing two objects, take the first, carry it straight to the target receptacle and place it, then fetch the second. Never take a second object while holding the first, and do not search while carrying the first for many steps.

#### L55: Both objects go in the same receptacle
Label: Decision
Description: Both objects must end in the same receptacle. Place the second at the same instance where the first completed the subgoal, not another receptacle of the same type.

#### L56: Second object must be a different instance
Label: Decision
Description: After placing one object, find a differently numbered instance of the target type. Do not re-take the one already placed, and leave it in the receptacle.

#### L57: Second instance may not exist
Label: Decision
Description: If every reachable receptacle has been searched and no unplaced second instance exists, it may be unreachable and the task may not be completable. Do not loop.

## Notes from discovery (ignored by the parser)

- Three lessons about the 'move' verb are kept separate because they give different behaviour: never use it, choose it as the normal placement form, and use it only as a fallback after put in/on fail.
- 'Object placed unprocessed cannot be retrieved' conflicts with the instruction in 'Put succeeded but task not complete' to take the object back from the wrong instance. They are kept separate because they describe different beliefs about whether an object can be taken out of a receptacle.
- 'Lamp use fails at a spot: try other surfaces' and 'Lamp use may be transiently ineffective' are kept separate: one says relocate, the other says retry and examine anyway.
- 'Stop retrying a broken appliance or lamp' is one lesson covering both devices; it sits in the processing theme but also applies to lamp tasks.
- Advice about heat failing because of the wrong held object or a faulty appliance is folded into 'Nothing happens means no effect' and the broken-device lesson.
- 'Always end with the action object' is kept as an Output lesson because the action is parsed from the response and a missing one cannot score.
- (validator) possible provenance wording 'earlier' in: A 'Nothing happens' response means the action had no effect, usually an unmet precondition (wrong location, object not h
