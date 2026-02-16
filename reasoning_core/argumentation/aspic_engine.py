"""
Layer 2: ASPIC+ Argumentation Knowledge Base.

ASPIC+ is a structured argumentation framework that captures:
- Strict rules (deductive: if premises, then conclusion)
- Defeasible rules (presumptive: normally, if premises then conclusion)
- Three attack types: rebut, undermine, undercut
- Preference orderings between arguments

Used to decompose philosophical arguments into verifiable components
and track how objections (attacks) relate to main arguments.

Reference: PHILOSOPHER_ENGINE_ARCHITECTURE.md, Layer 2
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
import json


class AttackType(Enum):
    """Three types of argumentative attack in ASPIC+."""
    REBUT = "rebut"          # Attacks the conclusion directly
    UNDERMINE = "undermine"  # Attacks a premise
    UNDERCUT = "undercut"    # Attacks the inference rule


class RuleType(Enum):
    """Rule strength in ASPIC+."""
    STRICT = "strict"           # Deductively valid
    DEFEASIBLE = "defeasible"   # Presumptively valid


@dataclass
class Rule:
    """An inference rule in the ASPIC+ knowledge base.

    Strict rules (→): if all premises hold, conclusion must hold.
    Defeasible rules (⇒): if all premises hold, conclusion presumably holds
    unless defeated.
    """
    rule_id: str
    premises: List[str]
    conclusion: str
    rule_type: RuleType = RuleType.STRICT
    label: str = ""
    source: str = ""       # e.g. "Meditation VI", "Fourth Objections"

    def __str__(self):
        arrow = "→" if self.rule_type == RuleType.STRICT else "⇒"
        prems = ", ".join(self.premises)
        return f"[{self.rule_id}] {prems} {arrow} {self.conclusion}"


@dataclass
class Attack:
    """An attack relation between arguments."""
    attacker_id: str
    target_id: str
    attack_type: AttackType
    target_component: str = ""   # Which premise or rule is attacked

    def __str__(self):
        return (f"{self.attacker_id} --{self.attack_type.value}-> "
                f"{self.target_id} ({self.target_component})")


@dataclass
class Argument:
    """A structured argument built from rules.

    An argument is a tree of rules leading to a conclusion.
    Each node is either an axiom premise or the conclusion
    of an applied rule.
    """
    arg_id: str
    conclusion: str
    sub_arguments: List['Argument'] = field(default_factory=list)
    top_rule: Optional[Rule] = None
    premises: List[str] = field(default_factory=list)
    attacks_received: List[Attack] = field(default_factory=list)
    attacks_given: List[Attack] = field(default_factory=list)

    @property
    def is_atomic(self) -> bool:
        """Atomic arguments have no sub-arguments (just premises)."""
        return len(self.sub_arguments) == 0

    @property
    def all_premises(self) -> Set[str]:
        """Collect all premises in the argument tree."""
        prems = set(self.premises)
        for sub in self.sub_arguments:
            prems.update(sub.all_premises)
        return prems

    @property
    def is_strict(self) -> bool:
        """Is this argument built entirely from strict rules?"""
        if self.top_rule and self.top_rule.rule_type != RuleType.STRICT:
            return False
        return all(sub.is_strict for sub in self.sub_arguments)


class ASPICKnowledgeBase:
    """ASPIC+ knowledge base for philosophical argumentation.

    Stores rules, builds arguments, and tracks attacks.
    Provides the structural backbone for the claim extractor
    and feeds formal claims to the Z3 verification engine.
    """

    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self.attacks: List[Attack] = []
        self.arguments: Dict[str, Argument] = {}
        self.preferences: Dict[Tuple[str, str], str] = {}  # (a, b) -> winner

    def add_rule(self, rule: Rule):
        """Add an inference rule to the knowledge base."""
        self.rules[rule.rule_id] = rule

    def add_attack(self, attack: Attack):
        """Register an attack between arguments."""
        self.attacks.append(attack)
        if attack.target_id in self.arguments:
            self.arguments[attack.target_id].attacks_received.append(attack)
        if attack.attacker_id in self.arguments:
            self.arguments[attack.attacker_id].attacks_given.append(attack)

    def add_preference(self, preferred: str, over: str):
        """Declare argument preference (for resolving attacks)."""
        self.preferences[(preferred, over)] = preferred

    def build_argument(self, arg_id: str, conclusion: str,
                       rule_ids: List[str],
                       premises: Optional[List[str]] = None) -> Argument:
        """Construct an argument from rules in the KB."""
        rules = [self.rules[rid] for rid in rule_ids if rid in self.rules]
        top_rule = rules[-1] if rules else None

        arg = Argument(
            arg_id=arg_id,
            conclusion=conclusion,
            top_rule=top_rule,
            premises=premises or [],
        )
        self.arguments[arg_id] = arg
        return arg

    def get_attacks_on(self, arg_id: str) -> List[Attack]:
        """Get all attacks targeting a specific argument."""
        return [a for a in self.attacks if a.target_id == arg_id]

    def get_attacks_by(self, arg_id: str) -> List[Attack]:
        """Get all attacks made by a specific argument."""
        return [a for a in self.attacks if a.attacker_id == arg_id]

    def is_defeated(self, arg_id: str) -> bool:
        """Check if an argument is defeated (attacked by preferred arg).

        An argument is defeated if:
        1. It is attacked, AND
        2. The attacker is preferred (or equal preference and attacker
           uses a strict rule while target uses defeasible)
        """
        attacks = self.get_attacks_on(arg_id)
        if not attacks:
            return False

        for attack in attacks:
            # Check if there's a preference for the attacker
            pair = (attack.attacker_id, arg_id)
            rev_pair = (arg_id, attack.attacker_id)

            if pair in self.preferences:
                return True  # Attacker preferred
            elif rev_pair in self.preferences:
                continue     # Target preferred, attack fails

            # No preference: strict beats defeasible
            attacker = self.arguments.get(attack.attacker_id)
            target = self.arguments.get(arg_id)
            if attacker and target:
                if attacker.is_strict and not target.is_strict:
                    return True

        return False

    def grounded_extensions(self) -> Set[str]:
        """Compute the grounded extension (most skeptical).

        The grounded extension contains all arguments that are:
        - Unattacked, OR
        - All their attackers are defeated

        This is the minimal credulous acceptance set.
        """
        accepted = set()
        defeated = set()
        changed = True

        while changed:
            changed = False
            for arg_id in self.arguments:
                if arg_id in accepted or arg_id in defeated:
                    continue

                attacks = self.get_attacks_on(arg_id)
                undefeated_attacks = [
                    a for a in attacks
                    if a.attacker_id not in defeated
                ]

                if not undefeated_attacks:
                    accepted.add(arg_id)
                    changed = True
                elif all(a.attacker_id in accepted
                         for a in undefeated_attacks):
                    defeated.add(arg_id)
                    changed = True

        return accepted

    # -- Decomposition helpers for major arguments --

    def load_cogito_argument(self):
        """Load the Cogito argument structure."""
        self.add_rule(Rule(
            "cogito_r1", ["Doubts(ego)"], "Thinks(ego)",
            RuleType.STRICT, "Doubting entails thinking",
            "Meditation II"
        ))
        self.add_rule(Rule(
            "cogito_r2", ["Thinks(ego)"], "Exists(ego)",
            RuleType.STRICT, "Thinking entails existing",
            "Meditation II"
        ))
        self.build_argument(
            "cogito", "Exists(ego)",
            ["cogito_r1", "cogito_r2"],
            premises=["Doubts(ego)"]
        )

    def load_real_distinction(self):
        """Load the Real Distinction argument structure."""
        self.add_rule(Rule(
            "rd_r1",
            ["CDP(mind_without_body)"],
            "Possible(mind_without_body)",
            RuleType.DEFEASIBLE,
            "Conceivability → Possibility (needs divine guarantee)",
            "Meditation VI"
        ))
        self.add_rule(Rule(
            "rd_r2",
            ["Possible(mind_without_body)"],
            "Distinct(mind, body)",
            RuleType.STRICT,
            "Separability → Real Distinction",
            "Meditation VI"
        ))
        self.build_argument(
            "real_distinction", "Distinct(mind, body)",
            ["rd_r1", "rd_r2"],
            premises=["CDP(mind_without_body)"]
        )

        # Arnauld's attack on the conceivability premise
        self.add_rule(Rule(
            "arnauld_r1",
            ["Right_triangle_conceivable_without_pythagorean"],
            "CDP_does_not_entail_possibility",
            RuleType.DEFEASIBLE,
            "Arnauld's triangle analogy",
            "Fourth Objections"
        ))
        self.build_argument(
            "arnauld_objection", "CDP_does_not_entail_possibility",
            ["arnauld_r1"],
            premises=["Right_triangle_conceivable_without_pythagorean"]
        )
        self.add_attack(Attack(
            "arnauld_objection", "real_distinction",
            AttackType.UNDERCUT, "rd_r1"
        ))

    def load_zombie_argument(self):
        """Load Chalmers' zombie argument (structurally parallel
        to Real Distinction)."""
        self.add_rule(Rule(
            "zombie_r1",
            ["Conceivable(physical_duplicate_without_consciousness)"],
            "Possible(zombie_world)",
            RuleType.DEFEASIBLE,
            "Zombie conceivability → possibility",
            "Chalmers 1996"
        ))
        self.add_rule(Rule(
            "zombie_r2",
            ["Possible(zombie_world)"],
            "Not(Physicalism)",
            RuleType.STRICT,
            "Zombie possibility refutes physicalism",
            "Chalmers 1996"
        ))
        self.build_argument(
            "zombie_argument", "Not(Physicalism)",
            ["zombie_r1", "zombie_r2"],
            premises=["Conceivable(physical_duplicate_without_consciousness)"]
        )

    def to_aif(self) -> Dict:
        """Export to AIF (Argument Interchange Format) for interop."""
        nodes = []
        edges = []

        for arg_id, arg in self.arguments.items():
            nodes.append({
                "nodeID": arg_id,
                "type": "I",  # Information node
                "text": arg.conclusion,
            })

        for attack in self.attacks:
            nodes.append({
                "nodeID": f"attack_{attack.attacker_id}_{attack.target_id}",
                "type": "CA",  # Conflict Application
                "text": attack.attack_type.value,
            })
            edges.append({
                "fromID": attack.attacker_id,
                "toID": f"attack_{attack.attacker_id}_{attack.target_id}",
            })
            edges.append({
                "fromID": f"attack_{attack.attacker_id}_{attack.target_id}",
                "toID": attack.target_id,
            })

        return {"nodes": nodes, "edges": edges}

    def __repr__(self):
        return (f"ASPICKnowledgeBase("
                f"rules={len(self.rules)}, "
                f"arguments={len(self.arguments)}, "
                f"attacks={len(self.attacks)})")
