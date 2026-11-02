# Morning report

**In short: Task 2 and Task 3 now both run from your real videos to the robot in simulation. Task 3 only works after moving positions to fit the robot. Neither has been tested on a new video kept apart from tuning, so that's the main thing left.**

## What's new

| Work | Result | Caveat |
|---|---|---|
| **Task 2 from your video** | Built the triangle, tower 0.2 cm from the video's spot | One video, one run, settings tuned on it |
| **Task 3 from your video, fitted** | Built and flipped; order and flip read from video | Tower moved 31 cm, glasses moved 9-33 cm |
| **Task 3 from your video, exact positions** | **Failed**, dropped in the twist | The flip can't work at that tower spot |
| **Flip recognised from video** | Correct on your 2 right-side videos | Only those 2 tested |
| **Your 2 old left-side videos** | **Refused** by the reader | One worked with the earlier reader version |
| **Robot, Task 2 areas** | 28 of 30 | Exact simulator positions, pinned pelvis |
| **Robot, Task 3 areas** | 28 of 30 | Exact simulator positions, pinned pelvis |

## Watch these

- `docs/videos/task2_video_to_robot.mp4`
- `docs/videos/task3_video_to_robot_fitted.mp4`

## What I need from you

1. **Record 2 new Task 2 takes,** the fresh test. Areas are in `docs/recording_views/task2_initial_area.png` and `task2_final_area.png`.
2. **Record 1 new Task 3 take,** so the robot can use your real positions. Areas are in `task3_initial_area.png` and `task3_final_area.png`.
3. **Fill in `docs/DATA_COLLECTION.md`:** the lines marked "to confirm".
4. **Tell me when to make the repo public,** then submit before Wed 4 Nov, 23:59 IST.

**Keep the camera and table where they were for `task2_triangle_right_take1`.** New takes then need no table corners. If either has moved, tell me, and I'll update the corners.

## Before going public

The repo will include your two sample videos, showing your body and feet, but not your face. Transcripts, papers and original videos stay out of git.
