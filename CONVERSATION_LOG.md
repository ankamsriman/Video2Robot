# Conversation log

A short record of how the project discussion went and why each decision was made.

---

## Session 1: Saturday, Oct 31, 2026

### Where we started
You shared `task.md`, two session transcripts and five papers, and asked for a discussion before any building. You also suggested a brain file plus a separate, nicely formatted record of our conversation. This is that record.

### What we found

| Finding | What it meant |
|---|---|
| `task.md` names a 6-DOF arm, but organisers prefer a humanoid | Use the Unitree G1 |
| The `../Demo2Robot` folder in `task.md` doesn't exist | Record our own videos |
| M1 Mac, 8 GB RAM, under 10 GB free disk, no NVIDIA GPU | MuJoCo on the Mac; heavy vision models on a GPU |
| Deadline mentioned as "November 4th or 5th" | Plan for Nov 4 until confirmed |
| Mentors gave conflicting camera advice | Fix the camera, so real positions can be measured |

### What we decided

| Question | Your answer | Effect |
|---|---|---|
| Deadline | Not sure yet | Plan for Nov 4 |
| Resources | Hackathon or cloud GPU; no API keys | GPU handles object tracking |
| Robot scope | Standing G1, then upgrade | Lower risk, meets the humanoid preference |
| Demo task | Sort objects into containers | SeeDo's strongest task type, and clear to show |
| Vision AI model | Discuss later | Decide at the start of Phase 4; planner stays swappable |
| Experience | New to Python and robotics | Plain-language explanations |
| Recording setup | Tripod, printer, table and objects | Printed marker is possible |
| Rules and team | Solo; only `task.md` so far | Submission format still to confirm |

### The idea that makes it stand out
SeeDo has three gaps, and we turn each into something measurable:

> 1. **Where things go:** the AI decides what moved, and measured positions decide where.
> 2. **Real distances:** a printed marker converts pixels into centimetres.
> 3. **Did it work?** The robot tries, the result is compared with your video, and it retries. This gives the mentors' own metric: attempts to success.

### What got built

**Phase 0 complete.** The project skeleton, Python environment and G1 model are in place. The G1 renders on your Mac and the first 2 tests pass.

**Phase 1 complete.** The G1 sorts three blocks in real physics, using friction only. It went from 0 of 10 to 20 of 20 on new setups, after a scoring fix explained below. The pelvis is pinned, because the robot falls on far reaches when unpinned.

### What changed along the way

| Moment | What happened | Decision |
|---|---|---|
| Official text | `task.md` turned out to hold a tracker; the official challenge text was pasted back | Plan from the official text. Deadline confirmed as Wed Nov 4 |
| Training | Post-training is the official example, but not mandatory | Do A, then B, then C, as time permits |
| Slow Mac | iCloud had removed local copies of project files | You downloaded them and cleared space |
| Project location | I suggested moving it out of Desktop | You kept it there, so the code survives losing the laptop |
| Scoring flaw | The 4 failures out of 20 had actually placed the object correctly; my check had judged the hand | Judge where the object ends up, then retest on new setups |
| Second task | You proposed a triangle of cups | Try it on a separate branch, so sorting stays safe |

### Overnight work
| Work | Result |
|---|---|
| Video to robot | Works on a rendered video, not yet on real video |
| Cup triangle, on branch `cup-task` | 30 of 30 with a written plan, and also works from a rendered video |
| Retry loop | Weak grip went from 0 of 8 to 8 of 8, after fixing a test that measured nothing |
| Working area | 70 of 78 positions passed, larger than first estimated |

### Task 3: the flipped cup
Your idea: a fourth cup on the triangle, opening up, with the robot turning it over. It went from 1 of 9 to 21 of 24 on new setups, on its own branch. You then asked for 100%: aiming at the measured top cup and cancelling the shift on release gave 50 of 50 on new setups. The flip is programmed, so it isn't learned from video.

### Still open
AI model and API key, GPU access, and which tasks to show in the final submission.
