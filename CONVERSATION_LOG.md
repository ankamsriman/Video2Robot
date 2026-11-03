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

### Picturing the recording view
You asked to see what the phone should see. Drawing it showed the blocks hiding the marker from a 75 cm tripod, so the marker moved to the phone's end of the table. It also exposed a tracker bug that put positions up to half a block off, now fixed. Pictures are in `docs/recording_views/`.

### The pipeline, explained step by step
You asked for the whole pipeline from scratch. `docs/PIPELINE_EXPLAINED.md` follows take 5 through every step. No AI model runs in this pipeline: it uses image processing and hand-written rules. Writing it showed that the corner fit is worse on takes since the camera moved.

### Still open
AI model and API key, GPU access, and which tasks to show in the final submission.

---

## Session 2: Monday, Nov 2, 2026

### Your real videos changed the plan
You recorded steel glasses with no marker, and with no blocks to hand. That ruled out colours and the marker, so the system now follows bright glass tops and uses the table as its ruler.

| Decision | Why |
|---|---|
| Tasks are Task 2 and Task 3, and real-video sorting is dropped | You have no blocks, and the triangle already meets the hard rule |
| No marker | The table's corners and measured size do the job |
| Identical glasses kept | Rest spots keep the order, so no SAM 2 or coloured cups |
| Task 3 can be recorded again in the flip's area | The arm can't flip across its own chest |
| Positions may be moved to fit the robot, always reported | Better than hiding the limit or giving up on Task 3 |
| Repo stays private until you say so | Your choice |

### Corrections along the way
The table was first given as 80 x 60 cm, and the video's shape disagreed, so it was measured again at 80 x 50. A rim of 1.3 cm was a voice-typing slip for 7.3 cm. I also wrongly said the robot always grips from the side: it grips the first three glasses from above.

### Task 3 no longer needs moved glasses
The fitted layout showed your video was a mirror image of what the right arm wants. So the robot now picks its arm and standing spot from the layout. On your original video, that's the left arm, 23.4 cm right of the table's middle, with no glass moved. The robot starts at that spot, since it can't walk.

### Task 2 uses the nearest arm
You asked for the robot to choose the arm nearest the glasses. It helps on left-side layouts (24 of 30 against 10 of 30), and not on layouts spread across both sides (20 against 21 of 30). It keeps one arm for the whole task.

### Eight new Task 2 takes
You recorded eight takes. The reader accepted 2 and the robot failed both, so none succeeded. I added a rule that works out an unseen top glass from the other glasses, and takes 2 and 5 then succeeded with the left arm. The rule came after seeing these videos, so they aren't a fair test of it. Takes 3, 4, 7 and 8 are still refused because a starting glass is missed. Take 4 and take 6 are mixed layouts one arm can't do, so they suit your two-arm idea.

### Six new Task 3 takes
You recorded six Task 3 takes in different layouts, and I ran them with nothing changed. None succeeded: the reader refused three, and the robot failed the other three. Updating the table's corners changed no result. We agreed the order: one more Task 3 video, make the submission safe, then improve Task 2 with your two-arm idea and a better reader.

### Task 3 works on a new video
Take 8 was refused, and the glass to flip was too far for the robot. Take 9 succeeded with nothing changed: left arm, tower 0.4 cm off. It's one run of a chosen layout, and that layout passed only 3 of 6 simulator runs when nudged.

### Your two-arm idea
Built as an optional mode, for Task 2 only. My first version scored 9 of 30, because the old arm knocked glasses during the swap. With the hands going around the glasses, it tied with one arm on random layouts (20 against 20, and 22 against 21 of 30). It clearly beat one arm on your take 4 and take 6 layouts, which were picked because one arm struggled, and it made the real take 6 video succeed. So it's off by default.

### Copying your hand's path
You pointed out that the video told the robot what to do, not how to move. So the hand tracker now follows your hand, and with `--hand-path` the robot carries each glass with your hand's height and timing. Takes 1, 2, 5 and 6 succeed. The grip and finger poses stay programmed, and nothing is trained.

### A vision language model, overnight
You added an API key and asked for a VLM. Claude now turns typed instructions into checked plans, chooses which glass to flip, and finds glasses the reader missed. Task 2 went from 4 to 8 of your 9 videos built, though I adjusted it on those same videos. On 10 cluttered simulated tables it found exactly the three glasses every time, within 1.5 cm. Grip timing was dropped: the tracker couldn't see the fingers around the glass.

### Still open
New videos to test the VLM fairly. Making the repo public, and submitting. The data collection note needs your details. You decide when the repo goes public.
