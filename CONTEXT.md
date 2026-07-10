AMD Developer Hackathon: Participant Submission Guide 
This document covers everything you need to build and submit a competitive entry. Exact evaluation inputs are intentionally omitted: your agent must be genuinely capable, not hardcoded to specific answers. 
Track 1: General-Purpose AI Agent 
What you are building 
An AI agent that handles a wide variety of natural language tasks across multiple capability domains, using Fireworks AI models as efficiently as possible. 
Why this task exists: 
Enterprises want to control AI spend without sacrificing user experience: not every task needs a premium proprietary model. A common pattern is hosting a range of models in-house (open-source, fine-tuned, RAG-based) and only calling out to a premium API when genuinely necessary. Track 1 asks you to build that smart router: run as many local models as you need, they cost zero toward your score, and make as few external Fireworks API calls as possible while still clearing the accuracy gate. 
Capability categories 
Your agent will be evaluated across all eight categories. Build for all of them:
# 
Category 
What it covers
1 
Factual knowledge 
Explaining concepts, 
definitions, and how things work
2 
Mathematical reasoning 
Multi-step arithmetic, 
percentages, word problems, projections
3 
Sentiment classification 
Labelling sentiment and justifying the classification



# 
Category 
What it covers
4 
Text summarisation 
Condensing passages to a specific format or length constraint
5 
Named entity recognition 
Extracting and labelling 
entities (person, org, location, date)
6 
Code debugging 
Identifying bugs in code 
snippets and providing 
corrected implementations
7 
Logical / deductive 
reasoning
Constraint-based puzzles where all conditions must be satisfied
8 
Code generation 
Writing correct, 
well-structured functions from a spec



What to submit 
A Docker image pushed to a public registry (e.g. GitHub Container Registry, Docker Hub). Check out the Image Architecture requirement at the bottom of this document 
Your container must: 
1. Read tasks from /input/tasks.json on startup 
JSON 
[ 
 { "task_id": "t1", "prompt": "Summarise the following text in one sentence: ..." }, 
 { "task_id": "t2", "prompt": "..." } 
] 
2. Write results to /output/results.json before exiting 
JSON 
[ 
 { "task_id": "t1", "answer": "..." },
 { "task_id": "t2", "answer": "..." } 
] 
Practice tasks (not the real evaluation set) 
These are illustrative examples only — not the real grading tasks (those stay hidden). Use them to validate your container's input/output handling locally before using a real submission slot. 
JSON 
[ 
 { "task_id": "practice-01", "prompt": "What is the capital of Australia, and what body of water is it near?" }, 
 { "task_id": "practice-02", "prompt": "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?" }, 
 { "task_id": "practice-03", "prompt": "Classify the sentiment of this review: The battery life is great, but the screen scratches too easily." },  { "task_id": "practice-04", "prompt": "Summarize the following in exactly one sentence: [your own sample paragraph here]." }, 
 { "task_id": "practice-05", "prompt": "Extract all named entities and their types from: Maria Sanchez joined Fireworks AI in Berlin last March." },  { "task_id": "practice-06", "prompt": "This function should return the max of a list but has a bug: def get_max(nums): return nums[0]. Find and fix it." },  { "task_id": "practice-07", "prompt": "Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?" }, 
 { "task_id": "practice-08", "prompt": "Write a Python function that returns the second-largest number in a list, handling duplicates correctly." } ] 
Environment variables: 
The harness injects these at runtime. Read them from the environment: do not hardcode values or bundle a .env file in your image. 
Python 
import os
api_key = os.environ["FIREWORKS_API_KEY"] # provided by harness — do not use your own 
base_url = os.environ["FIREWORKS_BASE_URL"] # route ALL Fireworks calls through this URL 
models = os.environ["ALLOWED_MODELS"].split(",") # exact model IDs published on launch day 
For local development you can use a .env file, but your submitted container must read these purely from the environment: the harness will inject the real values at evaluation time. 
Variable 
Description
FIREWORKS_API_KEY 
Provided by the harness — use this key, not your own
FIREWORKS_BASE_URL 
Base URL for all Fireworks API calls — must be used to configure your client
ALLOWED_MODELS 
Comma-separated list of permitted Fireworks AI model IDs, published on launch day



Important: All API calls must go through FIREWORKS_BASE_URL. Calls that bypass this URL will not be recorded and the submission will score zero tokens. Do not hardcode model IDs: read from ALLOWED_MODELS at runtime. 
Rules 
- Exit code 0 on success, non-zero on failure 
- Maximum runtime: 10 minutes 
- Only models in ALLOWED_MODELS are permitted, calls to other models invalidate the submission 
- /output/results.json must be valid JSON, malformed output scores zero - Local models and tokens used locally count as zero for the final score; all Fireworks API calls must go through FIREWORKS_BASE_URL; local model inference inside the container is permitted and counts toward accuracy, but not toward the token score. - Do not hardcode or cache answers; evaluation uses unseen prompt variants - Image compressed size must not exceed 10GB — larger images are rejected before pulling 
- Submissions are rate-limited to 10 per hour per team
- Grading environment: 4 GB RAM, 2 vCPU. If bundling a local model, size it to fit within these limits (2B–3B 4-bit quantized models are safe; 7B 4-bit fills the full RAM budget, leaving no room for agent code). 
Scoring 
1. Accuracy gate: LLM-Judge evaluates each answer against the expected intent. Submissions below the accuracy threshold are excluded from the leaderboard. 2. Token efficiency: submissions that pass the accuracy gate are ranked ascending by total tokens recorded by the judging proxy. Fewer tokens = higher rank. 
A note on token counting: The underlying task prompts are identical for every team, but your own system prompt (verbosity instructions, formatting requests, etc.) affects your input token count, and your model's response length affects output tokens. Don't over-optimize output length early, focus first on your routing logic and which local models you use. Output-length tuning is a good later-stage optimization once your router is solid. 
Troubleshooting: why did my submission fail? 
If your submission doesn't score as expected, here's what each status means and how to fix it. Most of these also apply to Track 2.
Status 
What it means & how to fix it
PULL_ERROR 
We couldn't pull your Docker image. Confirm it's public, and includes a linux/amd64 manifest (Apple Silicon builds need docker buildx build --platform linux/amd64).
RUNTIME_ERROR 
Your container ran but exited with a non-zero error code. Check your own container logs locally — something in your agent code crashed.
TIMEOUT 
OUTPUT_MISSING 
Your container didn't finish within the 10-minute limit. Check for hangs, infinite loops, or excessive retries in your agent. 
Your container exited cleanly but never wrote /output/results.json. Confirm your code writes this file before exiting.
INVALID_RESULTS_SCHEMA 
/output/results.json isn't in the right format. Each entry must be a JSON object with both





a task_id and an answer field.
MODEL_VIOLATION 
You called a Fireworks model that isn't in the published ALLOWED_MODELS list. Only call models from that list, read it from the env var at runtime, don't hardcode it.
IMAGE_TOO_LARGE 
Your image is over the 10 GB compressed size limit. Trim unnecessary 
layers/dependencies from your Docker image.
ACCURACY_GATE_FAILED 
Your container ran fine, but your answers scored below the accuracy threshold. This is a quality issue with your agent's answers, not an infrastructure problem.



Note: you may also see a flagged: ZERO_API_CALLS marker alongside your result; this is not a failure. It just means your submission made zero calls through the Fireworks proxy (e.g. a local-model-only agent), which is a valid strategy per the local models rule above. 

 Track 1: General-Purpose AI Agent Build an agent that handles tasks across 8 categories (factual Q&A, math reasoning, sentiment, summarization, NER, code debugging, logic puzzles, code generation) using Fireworks AI models. Submit a Docker image. Scored on an accuracy gate, then ranked by token efficiency. Allowed models (Track 1):
minimax-m3
kimi-k2p7-code
gemma-4-31b-it
gemma-4-26b-a4b-it
gemma-4-31b-it-nvfp4


FAQS
# 
Question 
Answer
1 
What is the hackathon 
website?
https://lablab.ai/ai-hackathons/amd-developer-hackath on-act-ii
2 
I did not get AMD GPU Cloud credits.
You do not need them for the hackathon, you will get a Jupyter Notebook cloud instance made available to you after hackathon start with periods of cloud access.
3 
How do I access hackathon compute?
"https://notebooks.amd.com/hackathon" 
( not "https://notebooks.amd.com") 
You will be directed to Sign into AMD Developer Program and provided access to your team’s compute resource. These are available on hackathon start.
4 
Where can I save things for persistent storage for the 
duration of the hackathon
/workspace
4 
How much persistent storage does a team get?
25 GB of persistent store. Please note that it takes a few minutes to flush your local store to persistent store when you request pod deletion.
5 
How much compute time does my team get
Each time will get a fixed number of hours over a 24-hour period. The time duration and reset interval (24 hours) will initially be longer but closer to hackathon termination, it will be shorter, to ensure everyone has access to the machines without a long wait.
6 
How can one quota 
consumption when not actually working?
Turn off your session, Pod. Everything in your persistent storage will remain intact. Once a teammate restarts your session, your compute time will start getting consumed. When multiple teammates use a session concurrently, there is not extra compute change. It is no different from one or more persons using a single server machine.
7 
I have not got my Fireworks Credits
If you signed up after July 2, you will be allocated credits on July 8.



8. 
When is the event submission deadline?
CET 6 pm July 12, 2026
9
When will be results become available?
Approximately a month for the Unicorn track. The other tracks have a leaderboard. All results will be announced together.
10 
How do I join a team? 
On the LabLab.ai Discord under AMD Hackathon Act-2 there is a sub-channel to request teammates. https://discordapp.com/channels/87705644895634640 8/1488590902661222600
11 
Which models may one use Fireworks AI?
MiniMax and Kimi K 
MiniMax is a series of advanced AI models designed for multimodal understanding, coding, and agentic reasoning, with capabilities spanning text, audio, image, video, and music. MiniMax M3 - Coding & Agentic Frontier, 1M Context, Multimodal | MiniMax 
Kimi K series, from Moonshot AI, is a line of open-source large language models designed for agentic intelligence, coding, and multimodal tasks. The models use a Mixture-of-Experts (MoE) architecture, allowing efficient activation of a subset of experts per token, which enables large-scale reasoning without excessive computational cost
12 
How much memory does the GPU on the hackathon 
compute instance have?
About 48 GB. Remember that things like KV-Cache need space too and other infrastructure, so your model needs to be smaller. Please experiment launching with any model you are interested in.
13 
How do I launch a model 
serving instance with vllm?
In an terminal window in your Jupyter pod launch: > vllm serve Qwen/Qwen2-7B-Instruct --port 8000 --gpu-memory-utilization 0.3



14 
How do I fine-tune a model that needs more training time than my compute window.
Please use checkpointing to save your training state in persistent store. One or two checkpoints should suffice to allow you to pick up where you left off. 
Saving 
checkpoint = { 
"epoch": epoch, 
"global_step": step, 
"model_state_dict": model.state_dict(), 
"optimizer_state_dict": optimizer.state_dict(), "loss": loss, 
} 
torch.save(checkpoint, "checkpoint.pt") 
Resuming: 
checkpoint = torch.load("checkpoint.pt") 
model.load_state_dict(checkpoint["model_state_dict"]) optimizer.load_state_dict(checkpoint["optimizer_state_ dict"]) 
start_epoch = checkpoint["epoch"] + 1 
step = checkpoint["global_step"]
15 
How do I access Track1 
Leaderboard?
https://lablab.ai/ai-hackathons/amd-developer-hackath on-act-ii/live?track=1#amd-leaderboard
16 
How do I access Track 2 
leaderboard?
https://lablab.ai/ai-hackathons/amd-developer-hackath on-act-ii/live?track=2#amd-leaderboard
17. 
Why did I get locked out of my AMD Developer Account while using the AMD AI Academy Jupyter Notebook? How do I get unlocked?
The ai-academy environment is to run notebooks cell-by-cell. If users experiment with new commands, the detection routines block the user. For experiments, they can explore AMD Dev Cloud or our hackathon platform for lablab hackathon. 
Please send us a support request. Unlocking could take anywhere between 1-3 days, 3 if spanning a weekend.



18 
What does N hours in 24 hours mean with respect to compute time?
It means your compute instance will be given resources on our GPU cluster for a maximum of N hours in a 24 hour window. You can use those N hours in one stretch or in multiple steps. To use in steps you need to “stop/shutdown” your instance and “restart” it later. Anyone on your team can shutdown or restart an instance, it is a shared instance. When you restart, if you are using a virtual environment, you will need to use pip install to install any libraries you want in your kernel. 
Occasionally, under low load we may reset the window to something shorter so you can hop on sooner or even give more hours. Typically close to the end of the hackathon when lots of people are seeking resources this does not happen!


Do I need approval for my idea for the hackathon?
No! Welcome! Mentors are a resource, not a gate.


How will I get my $50 
Fireworks AI credits?
You will receive an email with a code, one per team.


Do I need to maintain a live API endpoint for the judges for evaluation?
Absolutely not, it will just be too expensive. If you want you could set up a HuggingFace deployment of your application, which uses a serverless setup to avoid usage credits when not actually in use. For the purposes of hackathon submission you only need to create a demo video of your application, a 
presentation deck (about 5 slides) and provide your GitHub url.




Here's Track 1: Official clarification + guide updates Local models are a valid scoring strategy: Your container can answer tasks using a local model; those answers count fully toward accuracy. Only tokens routed through FIREWORKS_BASE_URL count toward your token score. A local model that answers a task correctly uses zero Fireworks tokens, the best possible outcome for ranking. Practical limits if you're using a local model:
Grading environment: 4 GB RAM, 2 vCPU: 2B–3B 4-bit quantized models fit comfortably; a 7B 4-bit model fills the full RAM budget, leaving no room for your agent code
No Ollama or model runtime is pre-installed: bundle model weights directly in your Docker image
Image size limit remains 10 GB compressed
Track 1 pipeline is now live and scoring normally. If your submission failed, it's due to a specific, fixable reason, which you can now find in the updated participant guide 📖 What's new in the participant guide:
Troubleshooting: why did my submission fail? (Track 1, after the Scoring section): a table explaining exactly what each failure status (PULL_ERROR, RUNTIME_ERROR, TIMEOUT, INVALID_RESULTS_SCHEMA, MODEL_VIOLATION, IMAGE_TOO_LARGE, ACCURACY_GATE_FAILED) means and how to fix it
Practice tasks (Track 1, after What to submit): illustrative example tasks (not the real grading set) so you can validate your container's input/output handling locally before using a real submission slot.
Sincere apologies for the confusion. 

Track 1 scoring
The accuracy gate is 80%: below that, you won't appear on top of the leaderboard regardless of token count.,
Identical percentages across teams (84.2%, 78.9%, 73.7%…) are just arithmetic; there are exactly 19 fixed tasks, so every score is n/19.,
Slightly different scores from the same code can happen: the LLM judge isn't perfectly deterministic run-to-run. Known tradeoff, not rigging.,
Tip while we clear the backlog: your registry's download counter (e.g., GitHub Packages, Docker Hub) shows whether we've pulled your image yet.

