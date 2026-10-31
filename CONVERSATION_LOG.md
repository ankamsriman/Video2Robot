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
