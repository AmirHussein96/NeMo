# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, GraphFst

try:
    import pynini
    from pynini.lib import pynutil

    delete_space = pynutil.delete(" ")

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


def get_quantity(decimal: 'pynini.FstLike', cardinal_up_to_hundred: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. 1 million -> integer_part: "one" quantity: "million"
    e.g. 1.5 million -> integer_part: "one" fractional_part: "five" quantity: "million"

    Args: 
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
    """
    numbers = cardinal_up_to_hundred
    suffix = pynini.union(
        "million",
        "billion",
        "trillion",
        "quadrillion",
        "quintillion",
        "sextillion",
        pynini.cross("M", "million"),
        pynini.cross("K", "thousand"),
        pynini.cross("B", "billion"),
    )
    res = (
        pynutil.insert("integer_part: \"")
        + numbers
        + pynutil.insert("\"")
        + pynini.closure(pynutil.delete(" "), 0, 1)
        + pynutil.insert("quantity: \"")
        + suffix
        + pynutil.insert("\"")
    )
    res |= (
        decimal
        + pynini.closure(pynutil.delete(" "), 0, 1)
        + pynutil.insert("quantity: \"")
        + (suffix | "thousand")
        + pynutil.insert("\"")
    )
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g. 
        -12.5006 billion -> decimal { negative: "true" integer_part: "12"  fractional_part: "five o o six" quantity: "billion" }
        1 billion -> decimal { integer_part: "one" quantity: "billion" }

    cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph
        cardinal_graph_hundred_component_at_least_one_none_zero_digit = (
            cardinal.graph_hundred_component_at_least_one_none_zero_digit
        )

        self.graph = cardinal.single_digits_graph.optimize()

        if not deterministic:
            self.graph = self.graph | cardinal_graph

        point = pynutil.delete(".")
        optional_graph_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        self.graph_fractional = pynutil.insert("fractional_part: \"") + self.graph + pynutil.insert("\"")
        self.graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        final_graph_wo_sign = (
            pynini.closure(self.graph_integer + pynutil.insert(" "), 0, 1)
            + point
            + pynutil.insert(" ")
            + self.graph_fractional
        )

        self.final_graph_wo_negative = final_graph_wo_sign | get_quantity(
            final_graph_wo_sign, cardinal_graph_hundred_component_at_least_one_none_zero_digit
        )

        # reduce options for non_deterministic and allow either "oh" or "zero", but not combination
        if not deterministic:
            no_oh_zero = pynini.difference(
                NEMO_SIGMA,
                (NEMO_SIGMA + "oh" + NEMO_SIGMA + "zero" + NEMO_SIGMA)
                | (NEMO_SIGMA + "zero" + NEMO_SIGMA + "oh" + NEMO_SIGMA),
            ).optimize()
            no_zero_oh = pynini.difference(
                NEMO_SIGMA, NEMO_SIGMA + pynini.accep("zero") + NEMO_SIGMA + pynini.accep("oh") + NEMO_SIGMA
            ).optimize()

            self.final_graph_wo_negative |= pynini.compose(
                self.final_graph_wo_negative,
                pynini.cdrewrite(
                    pynini.cross("integer_part: \"zero\"", "integer_part: \"oh\""), NEMO_SIGMA, NEMO_SIGMA, NEMO_SIGMA
                ),
            )
            self.final_graph_wo_negative = pynini.compose(self.final_graph_wo_negative, no_oh_zero).optimize()
            self.final_graph_wo_negative = pynini.compose(self.final_graph_wo_negative, no_zero_oh).optimize()

        final_graph = optional_graph_negative + self.final_graph_wo_negative

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
