# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import dummy_cut

from nemo.collections.speechlm2.data.s2s_st_dataset_concat_v1_1 import (
    _merge_short_supervisions,
    build_token_channel,
    collate_token_channel,
)

SR = 16000
FL = 0.08
PAD, BOS, EOS = 0, 1, 2


class _WordTokenizer:
    """Whitespace tokenizer: one token per word, plus single-token ``^`` and ``*``."""

    bos = BOS
    eos = EOS
    pad = PAD
    unk_id = None

    def __init__(self):
        self.vocab = {"^": 3, "*": 4}

    def text_to_ids(self, text):
        return [self.vocab.setdefault(w, len(self.vocab) + 3) for w in text.split()]


@pytest.fixture
def tokenizer():
    return _WordTokenizer()


def _sup(start, duration, text, speaker="agent"):
    return SupervisionSegment(id="s", recording_id="r", start=start, duration=duration, text=text, speaker=speaker)


def _cut(supervisions, duration):
    return dummy_cut(0, duration=duration, recording_duration=duration, supervisions=supervisions)


def _build(cut, tokenizer, min_sup_frames):
    return build_token_channel(
        cut, tokenizer, FL, roles={"agent"}, pad_id=PAD, min_sup_frames=min_sup_frames
    ).tolist()


def _words(tokenizer, tokens):
    inv = {v: k for k, v in tokenizer.vocab.items()}
    return [inv[t] for t in tokens if t in inv and t not in (3, 4)]


@pytest.mark.unit
def test_short_middle_supervision_is_merged_forward(tokenizer):
    # Mirrors cut _fr_mls_train_3060_2660_001645: the 0.15s agent segment "dans" is followed by a longer phrase.
    sups = [
        _sup(0.0, 2.35, "They create individualities", "user"),
        _sup(2.35, 2.17, "creent les individualites"),
        _sup(4.52, 0.15, "dans"),
        _sup(4.67, 2.23, "l'unite les proportions dans"),
        _sup(6.9, 1.99, "l'etendue l'innombrable"),
    ]
    cut = _cut(sups, duration=8.89)

    skipped = _build(cut, tokenizer, min_sup_frames=0)
    merged = _build(cut, tokenizer, min_sup_frames=3)

    # Without merging "dans" (1 frame) is dropped; with merging it leads the next phrase.
    assert _words(tokenizer, skipped).count("dans") == 1
    assert _words(tokenizer, merged).count("dans") == 2
    caret, star = tokenizer.vocab["^"], tokenizer.vocab["*"]
    ids = tokenizer.vocab
    start = merged.index(ids["dans"]) - 1
    assert merged[start] == caret
    assert merged[start + 1 : start + 4] == [ids["dans"], ids["l'unite"], ids["les"]]
    # Every written phrase has exactly one "^" and one "*": 3 agent phrases remain after the merge.
    assert merged.count(caret) == 3
    assert merged.count(star) == 3
    assert merged.count(BOS) == 1
    assert merged.count(EOS) == 1


@pytest.mark.unit
def test_short_last_supervision_is_merged_backward(tokenizer):
    cut = _cut([_sup(0.4, 1.6, "hello world"), _sup(2.0, 0.1, "again")], duration=2.1)

    tokens = _build(cut, tokenizer, min_sup_frames=3)

    assert _words(tokenizer, tokens) == ["hello", "world", "again"]
    assert tokens.count(tokenizer.vocab["^"]) == 1
    assert tokens.count(tokenizer.vocab["*"]) == 1
    assert tokens[-1] == EOS
    # With merging disabled the short last phrase is skipped instead.
    assert _words(tokenizer, _build(cut, tokenizer, min_sup_frames=0)) == ["hello", "world"]


@pytest.mark.unit
def test_consecutive_short_supervisions_cascade_forward(tokenizer):
    cut = _cut(
        [_sup(0.4, 1.0, "first phrase"), _sup(1.6, 0.08, "a"), _sup(1.68, 0.08, "b"), _sup(1.76, 1.0, "main phrase")],
        duration=2.8,
    )

    tokens = _build(cut, tokenizer, min_sup_frames=3)

    assert _words(tokenizer, tokens) == ["first", "phrase", "a", "b", "main", "phrase"]
    assert tokens.count(tokenizer.vocab["^"]) == 2
    assert tokens.count(tokenizer.vocab["*"]) == 2


@pytest.mark.unit
def test_lone_short_supervision_is_still_skipped(tokenizer):
    cut = _cut([_sup(0.4, 0.1, "tiny")], duration=1.0)

    tokens = _build(cut, tokenizer, min_sup_frames=3)

    assert set(tokens) == {PAD}


@pytest.mark.unit
def test_long_supervisions_are_unchanged(tokenizer):
    cut = _cut([_sup(0.4, 1.0, "one two"), _sup(1.4, 1.0, "three four")], duration=2.6)

    assert _build(cut, tokenizer, min_sup_frames=3) == _build(cut, tokenizer, min_sup_frames=0)


@pytest.mark.unit
def test_merge_helper_joins_text_span_and_keeps_input_intact():
    sups = [_sup(0.4, 1.0, "one"), _sup(1.4, 0.08, "short"), _sup(1.48, 1.0, "two")]

    merged = _merge_short_supervisions(sups, FL, SR, min_sup_frames=3)

    assert [s.text for s in merged] == ["one", "short two"]
    assert merged[1].start == pytest.approx(1.4)
    assert merged[1].end == pytest.approx(2.48)
    assert [s.text for s in sups] == ["one", "short", "two"]  # input not mutated
    assert _merge_short_supervisions(sups, FL, SR, min_sup_frames=0) is sups


@pytest.mark.unit
def test_collate_token_channel_does_not_mutate_cut_supervisions(tokenizer):
    cut = _cut([_sup(0.4, 1.0, "one"), _sup(1.4, 0.08, "short"), _sup(1.48, 1.0, "two")], duration=2.6)

    tokens, lens = collate_token_channel(
        CutSet([cut]), tokenizer, FL, roles={"agent"}, min_sup_frames=3
    )

    assert tokens.shape[0] == 1 and lens.tolist() == [tokens.shape[1]]
    assert [s.text for s in cut.supervisions] == ["one", "short", "two"]


@pytest.mark.skip(reason="token-channel stats counting is commented out in the dataset")
@pytest.mark.unit
def test_stats_count_truncated_and_skipped_phrases(tokenizer):
    # "one two three four five" needs 5 text frames but the 0.4s phrase only has room for 1.
    cut = _cut([_sup(0.4, 0.4, "one two three four five"), _sup(1.0, 1.0, "short ok")], duration=2.2)
    stats = {}

    build_token_channel(cut, tokenizer, FL, roles={"agent"}, pad_id=PAD, min_sup_frames=3, stats=stats)

    assert stats["phrases"] == 2
    assert stats["truncated"] == 1
    assert stats["dropped_tokens"] > 0
    assert stats.get("skipped", 0) == 0

    lone = {}
    build_token_channel(
        _cut([_sup(0.4, 0.1, "tiny")], duration=1.0), tokenizer, FL, roles={"agent"}, pad_id=PAD, stats=lone
    )
    assert lone == {"skipped": 1}
