"""Reading a video from SAM 2's glass tracks: each glass's own start, departure and end."""

import pytest

from src.perception.sam_reader import departure, final_point, spots_from_tracks

FPS = 30.0


def _track(start, end, left_frame, last=630, step=3):
    """A glass outline that sits at `start`, then sits at `end` from `left_frame` on."""
    return [
        [frame, *(start if frame < left_frame else end), 900]
        for frame in range(0, last + 1, step)
    ]


def test_a_glass_leaves_when_its_outline_moves_away_for_good() -> None:
    track = _track((400.0, 400.0), (800.0, 300.0), 270)

    assert departure(track, FPS, 40.0, 1.0) == pytest.approx(9.0)


def test_a_glass_whose_outline_never_moves_has_no_departure() -> None:
    assert departure(_track((400.0, 400.0), (400.0, 400.0), 0), FPS, 40.0, 1.0) is None


def test_a_brief_shift_is_not_a_departure() -> None:
    track = _track((400.0, 400.0), (400.0, 400.0), 0)
    track[50][1] = 600.0

    assert departure(track, FPS, 40.0, 1.0) is None


def test_the_end_is_where_the_outline_sits_over_the_last_second() -> None:
    assert final_point(_track((400.0, 400.0), (800.0, 300.0), 270), FPS) == (
        800.0,
        300.0,
    )


def test_tracks_become_starts_that_leave_in_order_and_ends_that_follow() -> None:
    tracks = {
        "0": _track((400.0, 400.0), (800.0, 300.0), 270),
        "1": _track((300.0, 500.0), (760.0, 420.0), 90),
        "2": _track((900.0, 520.0), (840.0, 420.0), 450),
    }
    claude_starts = [(401.0, 399.0), (299.0, 501.0), (900.0, 521.0)]

    spots, unused = spots_from_tracks(tracks, claude_starts, [], FPS, 40.0, 1.0, 30.0)

    starts, ends = spots[:3], spots[3:]
    assert [s.last for s in starts] == [90, 270, 450]
    assert (starts[0].x, starts[0].y) == (299.0, 501.0)
    assert [e.first for e in ends] == [91, 271, 451]
    assert unused == []


def test_a_track_that_never_moves_is_an_unused_glass() -> None:
    tracks = {
        "0": _track((400.0, 400.0), (800.0, 300.0), 270),
        "1": _track((484.0, 569.0), (484.0, 569.0), 0),
    }

    spots, unused = spots_from_tracks(tracks, [], [], FPS, 40.0, 1.0, 30.0)

    assert len(spots) == 2
    assert unused == [(484.0, 569.0)]


def test_ends_snap_to_claudes_more_exact_end_points_when_close() -> None:
    tracks = {"0": _track((400.0, 400.0), (800.0, 300.0), 270)}

    spots, _ = spots_from_tracks(tracks, [], [(812.0, 290.0)], FPS, 40.0, 1.0, 30.0)

    assert (spots[1].x, spots[1].y) == (812.0, 290.0)
