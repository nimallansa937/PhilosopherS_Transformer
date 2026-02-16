"""
Walton Argumentation Schemes for Philosophy of Mind.

Six scheme templates commonly used in philosophical arguments:
- Analogy: zombie ≈ Real Distinction
- Composition: physical parts → physical whole?
- Sign: neural correlates → consciousness?
- Expert Opinion: Descartes says X, therefore X
- Consequences: if dualism, then interaction problem
- Best Explanation: which theory best explains qualia?

Each scheme has:
- Premise template slots
- Critical questions that attackers can raise
- Mapping to ASPIC+ defeasible rules

Reference: PHILOSOPHER_ENGINE_ARCHITECTURE.md, Layer 2
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class CriticalQuestion:
    """A critical question that can defeat a scheme instance."""
    question_id: str
    text: str
    attack_type: str   # "undercut" or "undermine"
    description: str = ""


@dataclass
class WaltonScheme:
    """A Walton argumentation scheme template.

    Schemes are patterns of presumptive reasoning.
    They are defeasible — defeated if critical questions
    are answered negatively.
    """
    scheme_id: str
    name: str
    description: str
    premise_slots: List[str]
    conclusion_template: str
    critical_questions: List[CriticalQuestion] = field(default_factory=list)

    def instantiate(self, bindings: Dict[str, str]) -> Dict:
        """Fill in the scheme template with specific content.

        Args:
            bindings: Maps slot names to concrete content.
                e.g. {"case_1": "zombie argument",
                       "case_2": "Real Distinction"}

        Returns:
            Dict with filled premises and conclusion.
        """
        filled_premises = []
        for slot in self.premise_slots:
            filled = slot
            for key, val in bindings.items():
                filled = filled.replace(f"{{{key}}}", val)
            filled_premises.append(filled)

        filled_conclusion = self.conclusion_template
        for key, val in bindings.items():
            filled_conclusion = filled_conclusion.replace(
                f"{{{key}}}", val)

        return {
            "scheme": self.name,
            "premises": filled_premises,
            "conclusion": filled_conclusion,
            "critical_questions": [
                cq.text for cq in self.critical_questions
            ],
        }


# ============================================================
# SCHEME LIBRARY
# ============================================================

SCHEME_LIBRARY: Dict[str, WaltonScheme] = {}


def _register(scheme: WaltonScheme):
    SCHEME_LIBRARY[scheme.scheme_id] = scheme
    return scheme


# 1. Argument from Analogy
_register(WaltonScheme(
    scheme_id="analogy",
    name="Argument from Analogy",
    description=(
        "Two cases are similar in relevant respects, so what holds "
        "for one holds for the other."
    ),
    premise_slots=[
        "{case_1} has property {property}",
        "{case_1} is similar to {case_2} in respects {respects}",
    ],
    conclusion_template="{case_2} has property {property}",
    critical_questions=[
        CriticalQuestion(
            "analogy_cq1",
            "Are there relevant differences between {case_1} and {case_2}?",
            "undercut",
            "Disanalogy defeats the inference"
        ),
        CriticalQuestion(
            "analogy_cq2",
            "Is {property} transferable between the cases?",
            "undercut",
            "Some properties don't transfer across analogies"
        ),
        CriticalQuestion(
            "analogy_cq3",
            "Are the shared respects truly relevant to {property}?",
            "undermine",
            "Irrelevant similarities don't support the conclusion"
        ),
    ],
))

# 2. Argument from Composition
_register(WaltonScheme(
    scheme_id="composition",
    name="Argument from Composition",
    description=(
        "All parts have a property, so the whole has that property. "
        "Common in philosophy of mind: all brain parts are physical, "
        "so consciousness is physical?"
    ),
    premise_slots=[
        "All parts of {whole} have property {property}",
        "{whole} is composed of those parts",
    ],
    conclusion_template="{whole} has property {property}",
    critical_questions=[
        CriticalQuestion(
            "comp_cq1",
            "Can {property} be an emergent property not present in parts?",
            "undercut",
            "Emergence blocks composition inference"
        ),
        CriticalQuestion(
            "comp_cq2",
            "Does {property} apply to wholes the same way as parts?",
            "undermine",
            "Category error if property changes meaning at whole level"
        ),
    ],
))

# 3. Argument from Sign
_register(WaltonScheme(
    scheme_id="sign",
    name="Argument from Sign",
    description=(
        "Observable sign indicates underlying condition. "
        "E.g., neural correlate of consciousness (NCC) → consciousness."
    ),
    premise_slots=[
        "{sign} is observed",
        "{sign} is a sign of {condition}",
    ],
    conclusion_template="{condition} is present",
    critical_questions=[
        CriticalQuestion(
            "sign_cq1",
            "Is {sign} a reliable indicator of {condition}?",
            "undermine",
            "Correlation ≠ causation"
        ),
        CriticalQuestion(
            "sign_cq2",
            "Could {sign} be present without {condition}?",
            "undercut",
            "False positives undermine the inference"
        ),
    ],
))

# 4. Argument from Expert Opinion
_register(WaltonScheme(
    scheme_id="expert_opinion",
    name="Argument from Expert Opinion",
    description=(
        "Expert E says X, so X is (probably) true. "
        "Common in philosophical argument: Descartes argued X, "
        "Arnauld objected Y."
    ),
    premise_slots=[
        "{expert} is an expert in {domain}",
        "{expert} asserts that {claim}",
    ],
    conclusion_template="{claim} is (presumably) true",
    critical_questions=[
        CriticalQuestion(
            "expert_cq1",
            "Is {expert} truly an expert in {domain}?",
            "undermine",
        ),
        CriticalQuestion(
            "expert_cq2",
            "Do other experts in {domain} disagree?",
            "undercut",
        ),
        CriticalQuestion(
            "expert_cq3",
            "Is {expert} biased regarding {claim}?",
            "undercut",
        ),
    ],
))

# 5. Argument from Consequences
_register(WaltonScheme(
    scheme_id="consequences",
    name="Argument from Consequences",
    description=(
        "If theory T is true, consequence C follows. "
        "C is problematic/beneficial, so T is probably false/true. "
        "E.g., if substance dualism, then interaction problem."
    ),
    premise_slots=[
        "If {theory} is true, then {consequence} follows",
        "{consequence} is {evaluation}",  # problematic/beneficial
    ],
    conclusion_template="{theory} is (probably) {conclusion}",
    critical_questions=[
        CriticalQuestion(
            "conseq_cq1",
            "Does {consequence} really follow from {theory}?",
            "undermine",
        ),
        CriticalQuestion(
            "conseq_cq2",
            "Is {consequence} actually {evaluation}?",
            "undermine",
        ),
        CriticalQuestion(
            "conseq_cq3",
            "Are there countervailing consequences that outweigh?",
            "undercut",
        ),
    ],
))

# 6. Argument to Best Explanation (Abduction / IBE)
_register(WaltonScheme(
    scheme_id="best_explanation",
    name="Inference to Best Explanation",
    description=(
        "Among competing theories, the one that best explains "
        "the evidence is most likely true. Central to philosophy "
        "of mind debates about consciousness."
    ),
    premise_slots=[
        "{evidence} needs to be explained",
        "{theory} explains {evidence}",
        "No other theory explains {evidence} as well as {theory}",
    ],
    conclusion_template="{theory} is (probably) true",
    critical_questions=[
        CriticalQuestion(
            "ibe_cq1",
            "Does {theory} really explain {evidence} well?",
            "undermine",
        ),
        CriticalQuestion(
            "ibe_cq2",
            "Are there alternative explanations that are equally good?",
            "undercut",
        ),
        CriticalQuestion(
            "ibe_cq3",
            "What criteria define 'best' explanation here?",
            "undermine",
            "Simplicity? Scope? Predictive power?"
        ),
    ],
))
