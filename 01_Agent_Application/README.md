[🇬🇧 English](./README.md) | [🇨🇳 中文](./README_zh-CN.md)

<br>

## React Agent From Scratch

# learn-ai-agent

## Phase 1: API Communication & Basic Response Structure
- **Status**: Completed
- **Core Concept Comprehension**:
  1. `choices[0]`: `choices` is the list of candidate responses returned by the model (when the request parameter `n=1`, `[0]` represents the single generated response object).
  2. `finish_reason`: Indicates the model's response status. `'stop'` signifies that standard text generation has completed normally; if a tool invocation is required, the status shifts to `'tool_calls'`.
- **Troubleshooting Log**: Resolved Cloudflare gateway interception issues (fixed via network proxy/direct connection configuration).

## Phase 2: Under the Hood of Function Calling

### 1. Core Concepts & Q&A Synthesis

#### 1. Tool Matching & Decision Mechanism
* **Developer's Query**: If a tool's `description` is modified, will the LLM still invoke it?
* **Correct Understanding**: **No**. The LLM utilizes the Transformer attention mechanism to semantically match the user's prompt against `tools.description`. If the modified description no longer aligns semantically, the model will abandon the tool invocation and fallback to standard chat mode.
* **Edge Case Note**: Passing `tools` does **not** guarantee an intermediate invocation process. If the model determines its internal knowledge is sufficient to answer (e.g., "What is 1+1?"), it outputs text directly in the initial request (`finish_reason='stop'`), bypassing external execution.

#### 2. Statelessness & Dialogue Chaining
* **Developer's Query**: Are seemingly continuous tool-calling dialogues actually entirely independent HTTP requests from the LLM's perspective?
* **Correct Understanding**: **Absolutely correct**. The LLM itself lacks long-term memory; it is strictly stateless.
  - **Request 1 (Decision)**: Send (Prompt + Tool Schemas) $\rightarrow$ Model outputs the `tool_calls` instruction (with a unique `tool_call_id`), and cloud-side compute terminates and resets.
  - **Local Execution**: Python code parses the instruction and executes the actual function locally to retrieve the result.
  - **Request 2 (Synthesis)**: Python packages `[1. User Prompt, 2. Model Instruction, 3. Local Result]` as a **brand-new request** and feeds it to the LLM $\rightarrow$ Model reads the full history and synthesizes a natural language response.
* **Core Insight**: The so-called "continuous dialogue" is essentially local Python code continuously repackaging and transmitting the entire historical context back to the LLM via the `messages` array.

#### 3. Data Structures & Invocation Patterns
* **Message Format**: In `{"role": "xxx", "content": "yyy"}`, the `role` identifies the entity (`user` / `assistant` / `tool`), and the `content` holds the actual text payload. When the model issues a tool instruction, `content` can be null; the critical payload is housed within the `tool_calls` structure.
* **Parallel vs. Sequential**: LLMs support parallel tool calling (issuing multiple `tool_calls` instructions simultaneously in a single request); sequential multi-turn intermediate processes only occur when a subsequent step strictly depends on the output of a previous one.

  - **Issue Encountered:**
    How does it determine that my initial dialogue request requires tool invocation? How is the system's tool conversation automatically sent to it? What is the underlying logic?
    1. The answer is that the LLM matches my tool's description. (So, if I modify the description of this tool, does it mean the LLM will skip invoking it?)

Does modifying this tool's description prevent the LLM from invoking it?
Does this mean the seemingly continuous two-part dialogue is actually separated from the LLM's perspective? The first part is the prompt, then the LLM replies to the system to invoke a tool, and at that point, the session is over for the LLM. After the Python program executes, a second dialogue is initiated, packing the initial prompt along with the local execution result, allowing the LLM to combine both to generate the final answer.

Does passing multiple tools in a single prompt guarantee an intermediate process for the LLM every time? i.e., returning a session with `"finish_reason": "tool_calls"`. And only the final turn uses all the returned tool results and the prior user prompt, matching the tool descriptions to yield an answer?

What I mean is, for tools with various distinct functions, does the model determine which one to invoke based on the description? Or does it blindly invoke them all, and later rely on the descriptions to generate the correct answer during the synthesis phase?

''' Objective Fact: Description matching occurs at the starting point of every decision, not at the end point of the synthesized answer.

Starting Point (Decision Phase): The model reads the semantics of `tools.description` and decides: "Should I invoke a tool? If so, which one?"

End Point (Synthesis Phase): When the LLM receives all local execution results where `role='tool'`, its final synthesis heavily relies on the specific data payload returned in the `messages` history. At this stage, it no longer needs to re-validate the description; its sole objective is to format the acquired data into coherent language for the user. '''

I haven't verified this yet, but the logic should hold. For instance, if I define a calculator tool and ask "What is A+B?", the model invokes the tool once, resulting in two API calls in total. However, if I ask "What is the value of A+B, plus C?", there will be three API calls. The model first invokes the tool to get A+B, then invokes it again to add C to that value.

## Phase 3: ReAct Automated Loop Engine (Agent Engine)

### 1. Core Mechanisms & Architectural Design
1. **Automated `while` Loop Drive**: Uses `finish_reason` as the state-switching trigger:
   - `tool_calls`: Parse instruction $\rightarrow$ Execute local function $\rightarrow$ Append `role='tool'` record $\rightarrow$ Automatically trigger the next request.
   - `stop`: Decision converges, extract final response, exit the loop.
2. **Decoupling Routing (`TOOL_MAP`)**: Utilizes a dictionary to map string function names to actual Python functions. Combined with `**func_args` dictionary unpacking, this ensures zero modifications to the core engine when extending new tools.
3. **Engineering Circuit Breaker (`max_steps`)**: Explicitly sets a maximum loop threshold (e.g., `max_steps=5`) to prevent infinite loops caused by prompt ambiguity or logical exceptions.

### 2. Parallel & Sequential Tool Calling
- **Parallel Calling (Resolved in 1 Loop)**: When sub-tasks are independent (e.g., "check time + perform calculation"), the model issues multiple instructions simultaneously in a single `tool_calls` list. The loop executes all tools and submits the results to the model in one go.
- **Sequential Calling (Driven by Multi-Turn Loops)**: When a dependency chain exists between sub-tasks (e.g., `(A+B)+C`), the model utilizes multiple `while` loop iterations to progressively fetch intermediate results and advance its reasoning.

## Phase 4: External Memory & RAG Retrieval (Agent + RAG)

### 1. Core Concepts & Distinctions
1. **Traditional RAG vs. Agent+RAG**: Traditional RAG strictly relies on rigid, upfront interception and injection. In contrast, the Agent architecture encapsulates retrieval as a standard Tool, allowing the model to autonomously decide "whether to retrieve" and "what keywords to search for" based on user intent.
2. **Embedding Vectorization**: Maps raw text to high-dimensional spatial coordinates. By calculating coordinate distances to evaluate semantic similarity, it locates relevant local knowledge without requiring exact literal matches.

### 2. Under the Hood of HTTP Payloads & Parameter Extraction Mechanisms
1. **Full Request Payload**: LLMs are entirely stateless. With every HTTP request, Python must fully repackage and transmit the `model` configuration, the `tools` schemas, and the comprehensively chained `messages` dialogue history.
2. **Full Response Payload**: The API returns a complete JSON object containing the status code `finish_reason`, token `usage`, and the actual `message` payload. The code must manually unpack this and append it back to the local `messages` array.
3. **Intelligent Parameter Extraction**: Relying on top-level `description` directives in the JSON Schema (e.g., "extract core keywords") and the semantic constraints of the parameters themselves (e.g., "keywords"), combined with pre-trained common sense, the LLM automatically refines and transforms verbose natural language into structured API inputs.

## Phase 5: Engineering Wrap-Up & Underlying Mental Models (Agent Design Philosophy)

### 1. Dynamic Routing & Decoupling
- **High Cohesion, Low Coupling**: The Agent engine (responsible for dispatching requests and state transitions) and the Tool library (responsible for business logic like checking weather or math calculations) must be strictly decoupled.
- **`TOOL_MAP` Mapping**: Utilizes a dictionary to map the `func_name` string returned by the LLM to the actual Python function object (`TOOL_MAP[func_name]`).
- **New Tool Addition Principle**: No matter how many tools are added in the future, developers only need to follow a three-step process: "Write function $\rightarrow$ Add Schema $\rightarrow$ Register in TOOL_MAP". The core `while` loop engine requires **zero modifications** (adhering to the Open-Closed Principle, OCP).

### 2. Strict Dictionary Unpacking (`**func_args`)
- Python's `**` syntax instantaneously unpacks a JSON-parsed dictionary into keyword arguments for a function (e.g., `{"a": 1, "b": 2}` $\rightarrow$ `a=1, b=2`).
- **Fault Tolerance Boundary**: Unpacking matching is extremely strict. The dictionary Keys must identically match the parameter names defined in the Python function, otherwise a `TypeError` is thrown. Consequently, the parameter JSON Schema definitions within `tools` must be hyper-precise to guarantee the LLM outputs the correct Keys.

### 3. De-LangChain-ification & Statelessness
- **Shattering the Black Box**: Discard heavy frameworks in favor of native `list`, `while`, and HTTP SDKs to manually write the loop, securing 100% control over every Request/Response Payload.
- **State Camouflage**: Cloud-side LLMs are entirely stateless (fire-and-forget). The so-called "continuous multi-turn dialogue and tool invocation" is, in essence, the client forcibly maintaining context state. This is achieved by maintaining an ever-expanding `messages` array locally, continuously appending history (User, Assistant, Tool), and **fully retransmitting** the payload with every HTTP request.


## Phase 6: Real-World Development & Autonomous Sequential Chaining
- **Status**: Completed
- **Core Concept Comprehension**:
  1. **Environmental Reconnaissance & Autonomous Planning**: LLMs strictly avoid "blind guessing" missing required parameters. By introducing the `list_directory` tool, the Agent is granted the capability to "observe" its environment.
  2. **Multi-Turn Decision Chain (Planning)**: When presented with an ambiguous user prompt, the Agent autonomously initiates multiple rounds of API interactions (scout directory $\rightarrow$ identify target file $\rightarrow$ read contents $\rightarrow$ synthesize), successfully transitioning from "single-step instruction execution" to "workflow planning".