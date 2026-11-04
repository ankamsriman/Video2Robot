# Progress report

**In short: Task A and Task B both run from your real videos to the robot in simulation, and no glass has to move any more. Neither has been tested on a new video kept apart from tuning, so that's the main thing left.**

## Results

| Work | Result | Caveat |
|---|---|---|
| **Task A from your video** | Built the triangle, tower 0.2 cm from the video's spot | One video, one run, settings tuned on it |
| **Task B from your video** | Built and flipped, tower 0.5 cm from the video's spot, no glass moved | One video, one run. Left arm, robot starts 23.4 cm right of the table's middle |
| **Task B, robot kept mid-table** | **Failed**, dropped in the twist | Why the arm and standing spot are chosen |
| **Task A, nearest arm** | 24 of 30 against 10 of 30 for the right arm, on left-side layouts | No gain when spread across both sides, 20 against 21 of 30 |
| **Task B layout, wobbled up to 1.5 cm** | 19 of 21 | Exact simulator positions, pinned pelvis |
| **Random layouts, robot chooses** | 25 of 30 | Glass to flip 23-33 cm from your edge. 12 of 30 without that rule |
| **Flip recognised from video** | Correct on your 2 right-side videos | Only those 2 tested |
| **Your 2 old left-side videos** | **Refused** by the reader | One worked with the earlier reader version |
| **Robot, Task A areas** | 28 of 30 | Exact simulator positions, pinned pelvis |
| **Robot, Task B areas, mid-table** | 28 of 30 | Exact simulator positions, pinned pelvis |

## Watch these

- `docs/videos/task2_video_to_robot.mp4`
- `docs/videos/task3_video_to_robot.mp4`

## What I need from you

1. **Record 2 new Task A takes,** the fresh test. Areas are in `docs/recording_views/task2_initial_area.png` and `task2_final_area.png`.
2. **Record 1 new Task B take.** Use a different layout from your first, following "Steel glasses" in `docs/RECORDING_GUIDE.md`. The glass to flip must start at least 23 cm from your edge.
3. **Fill in `docs/DATA_COLLECTION.md`:** the lines marked "to confirm".
4. **Tell me when to make the repo public,** then submit before Wed 4 Nov, 23:59 IST.

**Keep the camera and table where they were for `task2_triangle_right_take1`.** New takes then need no table corners. If either has moved, tell me, and I'll update the corners.

## Before going public

The repo will include your two sample videos, showing your body and feet, but not your face. Transcripts, papers and original videos stay out of git.
