Subject: Sampling from LoRA samplers hangs indefinitely on our account (base model works) — request to release hot sampler weights

Hi Tinker team,

Since about 19:50 PT on 2026-09-18, every sampling request against a LoRA sampler on our account hangs with no
error and no billing, while sampling from the base model (Qwen/Qwen3.5-4B) works (~20 s for one token).
Example: a one-token sample from tinker://4ca87931-b181-562d-b41d-6e995779c1b3:train:0/sampler_weights/000010
times out after 60 s at 20:57, 21:02 (from a new project, 2e3a02db-674e-402e-8fdb-411f07b31d5c) and 21:07 PT.
Metadata endpoints answer normally; status page is green; no 402/429 in our logs.

What we think happened: an RL job that saves sampler weights every step ran with several concurrent copies today,
and several of those jobs were killed abruptly (Modal container stops and one preemption), leaving ~130 sessions
"Active" with in-flight sample requests and many sampler weight sets loaded. We have since finished all sessions
via POST /sessions/{id}/finish (accepted), but LoRA sampling is still paused. We believe we are hitting the
concurrent-sampler-weights limit and that the orphaned samplers are not being released.

Free metadata (retrieve_futures cursors on the finished sessions' samplers) shows the orphaned requests are still being
processed at roughly one completion every few minutes, i.e. hours of backlog ahead of any new request. The account
today: 160 sessions, 234 samplers, 40 training runs; 153 sessions are now client-finished.

Could you (0) drop the queued / in-flight sampling requests belonging to sessions that are already finished on our
organization (that alone would unblock us), (1) release / unload the sampler weights held by finished sessions, (2) tell us the
per-account limit on hot sampler weights and its eviction policy, and (3) confirm whether abandoned in-flight
sampling requests are cancelled when a session is finished or when its heartbeat lapses?

Account: (redacted; email and org id are in the support ticket).
SDK tinker 0.29.1, tinker-cookbook 0.5.7. Happy to share session ids and logs.

Thanks,
