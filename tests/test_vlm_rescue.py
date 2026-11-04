"""Adding glasses the reader missed, from points a vision model gives."""

import numpy as np

from src.perception.glass_tracker import Spot
from src.perception.vlm_rescue import (
    arrival_frame,
    departure_frame,
    departure_from_order,
    drop_clutter,
    drop_misread_ends,
    missing_points,
    rescued_spot,
    seen_from_start,
    trust_model_departures,
    unseen_ends,
    unused_glasses,
)


def _spot(x: float, y: float, first: int = 0, last: int = 100) -> Spot:
    return Spot(x, y, first, last, last - first + 1)


def test_a_point_near_a_spot_the_reader_found_is_not_missing() -> None:
    spots = [_spot(100, 100), _spot(400, 300)]
    points = [(103.0, 98.0), (700.0, 500.0)]

    assert missing_points(
        points, spots, fps=30.0, early_seconds=1.0, near_pixels=30
    ) == [(700.0, 500.0)]


def test_a_glass_the_reader_noticed_late_is_not_missing() -> None:
    spots = [_spot(700, 500, first=51, last=90)]

    assert (
        missing_points(
            [(700.0, 500.0)], spots, fps=30.0, early_seconds=1.0, near_pixels=30
        )
        == []
    )


def test_a_glass_noticed_late_is_moved_back_to_the_start() -> None:
    spots = [_spot(700, 500, first=51, last=90), _spot(100, 100, first=0, last=200)]

    updated = seen_from_start(
        [(700.0, 500.0)], spots, fps=30.0, early_seconds=1.0, near_pixels=30
    )

    assert updated[0].first == 0
    assert updated[1] == spots[1]


def test_the_glass_leaves_at_the_last_frame_its_patch_still_matches() -> None:
    change = np.concatenate(
        (np.full(60, 2.0), np.full(10, 40.0), np.full(20, 3.0), np.full(90, 35.0))
    )

    left = departure_frame(change, threshold=15.0, fps=30.0, settle_seconds=1.0)

    assert left is not None and abs(left - 89) <= 8


def test_one_noisy_frame_late_on_does_not_count_as_the_glass_still_there() -> None:
    change = np.concatenate((np.full(180, 10.0), np.full(150, 85.0)))
    change[300] = 5.0

    assert (
        abs(departure_frame(change, threshold=45.0, fps=30.0, settle_seconds=1.0) - 179)
        <= 8
    )


def test_a_glass_that_never_leaves_has_no_departure() -> None:
    assert (
        departure_frame(np.full(100, 2.0), threshold=15.0, fps=30.0, settle_seconds=1.0)
        is None
    )


def test_a_rescued_spot_runs_from_the_start_to_its_departure() -> None:
    spot = rescued_spot((700.0, 500.0), 89)

    assert (spot.x, spot.y, spot.first, spot.last) == (700.0, 500.0, 0, 89)


FPS = 30.0


def test_bright_clutter_there_from_the_start_is_dropped() -> None:
    # An earbud case sits 90 pixels from the nearest glass the model saw.
    earbuds = Spot(340.0, 581.0, 0, 690, 690)
    glass = Spot(400.0, 650.0, 0, 120, 120)

    kept = drop_clutter((earbuds, glass), [(402.0, 655.0)], FPS, 1.0, 30.0)

    assert kept == (glass,)


def test_spots_that_appear_later_are_kept_even_far_from_the_model() -> None:
    # Where the person put a glass down: not there when the model looked.
    placed = Spot(866.0, 408.0, 140, 690, 550)

    assert drop_clutter((placed,), [(100.0, 100.0)], FPS, 1.0, 30.0) == (placed,)


def test_without_model_points_nothing_is_dropped() -> None:
    spot = Spot(340.0, 581.0, 0, 690, 690)

    assert drop_clutter((spot,), [], FPS, 1.0, 30.0) == (spot,)


def test_clutter_uncovered_before_any_glass_left_is_dropped() -> None:
    # A box first seen at 1.2 s, once a hand moved away, before the first glass left at 3.4 s.
    glass = Spot(439.0, 441.0, 0, 102, 102)
    box = Spot(1127.0, 580.0, 36, 648, 129)

    kept = drop_clutter((glass, box), [(443.0, 439.0)], FPS, 1.0, 30.0)

    assert kept == (glass,)


LAST = 700


def test_a_spot_left_at_the_end_far_from_every_glass_the_model_saw_is_dropped() -> None:
    lower = Spot(920.0, 456.0, 150, LAST, 550)
    reflection = Spot(1200.0, 300.0, 400, LAST, 300)

    kept = drop_misread_ends(
        (lower, reflection), [(921.0, 455.0)], LAST, FPS, 1.0, 30.0
    )

    assert kept == (lower,)


def test_spots_that_left_before_the_end_are_not_judged_as_ends() -> None:
    start = Spot(440.0, 440.0, 0, 100, 100)

    assert drop_misread_ends((start,), [(900.0, 450.0)], LAST, FPS, 1.0, 30.0) == (
        start,
    )


def test_a_glass_the_model_saw_at_the_end_but_the_reader_missed_is_unseen() -> None:
    lower = Spot(920.0, 456.0, 150, LAST, 550)
    top = (862.0, 350.0)

    missing = unseen_ends((lower,), [(921.0, 455.0), top], LAST, FPS, 1.0, 30.0)

    assert missing == [top]


def test_a_glass_arrives_when_its_patch_starts_to_match_the_end_for_good() -> None:
    change = np.array([80.0] * 300 + [5.0] * 400)

    arrived = arrival_frame(change, 45.0, FPS, 1.0)

    # Smoothing over half a second blurs the moment by up to a quarter of a second.
    assert arrived is not None
    assert abs(arrived - 300) <= FPS / 4


def test_a_patch_that_only_matches_briefly_has_not_arrived() -> None:
    change = np.array([80.0] * 690 + [5.0] * 10)

    assert arrival_frame(change, 45.0, FPS, 1.0) is None


def test_a_glass_moved_first_left_before_the_first_landing() -> None:
    # Two glasses were seen leaving at 9.0 s and 16.0 s, but a glass landed at 4.7 s:
    # only the glass whose departure wasn't seen can have made that move.
    assert departure_from_order([270, 480], [141, 324, 600]) == 140


def test_a_glass_moved_last_left_before_the_last_landing() -> None:
    assert departure_from_order([102, 291], [150, 378, 600]) == 599


def test_no_unexplained_landing_means_no_departure_can_be_worked_out() -> None:
    assert departure_from_order([102, 291, 400], [150, 378, 600]) is None


def test_a_glass_the_reader_saw_arrive_but_lost_before_the_end_is_not_unseen() -> None:
    # Take 1: the top glass was seen from 23.9 s to 26.7 s, and the video ends at 29.3 s.
    lower = Spot(900.0, 511.0, 219, 879, 659)
    top_seen_then_lost = Spot(842.0, 377.0, 717, 801, 84)

    missing = unseen_ends(
        (lower, top_seen_then_lost),
        [(911.0, 506.0), (838.0, 375.0)],
        879,
        FPS,
        1.0,
        30.0,
    )

    assert missing == []


def test_a_glass_seen_in_the_same_place_at_both_ends_and_never_leaving_is_unused() -> (
    None
):
    # Hard take 1: the front-left glass stays put; its bright top is seen all video.
    stays = Spot(484.0, 569.0, 0, 646, 646)
    moved = Spot(496.0, 419.0, 6, 45, 40)
    start = [(496.0, 419.0), (484.0, 569.0)]
    end = [(785.0, 263.0), (480.0, 561.0)]

    unused = unused_glasses(start, end, (stays, moved), [45, None], 646, FPS, 1.0, 30.0)

    assert unused == [(484.0, 569.0)]


def test_an_unseen_glass_whose_patch_never_changed_is_unused() -> None:
    # Hard take 2: a glass the right way up shows no bright top, so the reader has no spot.
    start = [(994.0, 518.0)]
    end = [(990.0, 515.0)]

    assert unused_glasses(start, end, (), [None], 588, FPS, 1.0, 30.0) == [
        (994.0, 518.0)
    ]


def test_a_glass_that_left_is_not_unused_even_if_the_tower_ends_where_it_stood() -> (
    None
):
    left = Spot(500.0, 400.0, 0, 120, 120)

    assert (
        unused_glasses(
            [(500.0, 400.0)], [(502.0, 401.0)], (left,), [120], 588, FPS, 1.0, 30.0
        )
        == []
    )


def test_a_glass_the_model_sees_gone_at_the_end_left_even_if_something_bright_stays() -> (
    None
):
    # Hard take 1: a white tube beside the glass kept the reader's spot alive after it left.
    beside_tube = Spot(348.0, 514.0, 0, 646, 646)

    spots, unknown = trust_model_departures(
        (beside_tube,), [(350.0, 520.0)], [(786.0, 263.0)], [450], 646, FPS, 1.0, 30.0
    )

    assert spots[0].last == 450
    assert unknown == []


def test_without_a_measured_departure_the_glass_is_left_to_the_order_rule() -> None:
    beside_tube = Spot(348.0, 514.0, 0, 646, 646)

    spots, unknown = trust_model_departures(
        (beside_tube,), [(350.0, 520.0)], [(786.0, 263.0)], [None], 646, FPS, 1.0, 30.0
    )

    assert spots == ()
    assert unknown == [(350.0, 520.0)]


def test_a_glass_the_model_still_sees_at_the_end_is_left_alone() -> None:
    stays = Spot(484.0, 569.0, 0, 646, 646)

    spots, unknown = trust_model_departures(
        (stays,), [(484.0, 569.0)], [(480.0, 561.0)], [None], 646, FPS, 1.0, 30.0
    )

    assert spots == (stays,)
    assert unknown == []


def test_a_spot_that_comes_and_goes_mid_video_away_from_every_end_glass_is_dropped() -> (
    None
):
    # Hard take 3: the steel laptop stand shone from 3.9 s to 14.6 s, then stopped.
    lower = Spot(843.0, 403.0, 147, 700, 553)
    stand_glint = Spot(1290.0, 450.0, 117, 438, 200)

    kept = drop_misread_ends(
        (lower, stand_glint), [(845.0, 400.0)], 716, FPS, 1.0, 30.0
    )

    assert kept == (lower,)


def test_a_start_first_noticed_late_still_sets_when_the_first_glass_left() -> None:
    # Task 3 take 1: the first glass to move was first noticed at 2.9 s and left at 4.8 s,
    # so a glass put down at 9.7 s is a landing, not clutter.
    first_mover = Spot(533.0, 424.0, 87, 144, 58)
    other_start = Spot(431.0, 483.0, 0, 468, 469)
    landed = Spot(909.0, 496.0, 291, 1356, 1066)

    kept = drop_clutter(
        (first_mover, other_start, landed),
        [(539.0, 423.0), (431.0, 486.0)],
        FPS,
        1.0,
        30.0,
    )

    assert landed in kept


def test_a_landing_where_a_stacked_glass_belongs_is_kept() -> None:
    # Task 3: the triangle's top glass lands at 14.1 s and is covered at 20.8 s by the
    # flipped glass, so the model's end view can't see it. Its place, centred above two
    # glasses the model sees, exempts it.
    covered_top = Spot(1385.0, 168.0, 423, 624, 202)

    kept = drop_misread_ends(
        (covered_top,), [(682.0, 413.0)], 816, FPS, 1.0, 30.0,
        exempt=lambda spot: spot is covered_top,
    )  # fmt: skip

    assert kept == (covered_top,)
