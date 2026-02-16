"""
Layer 1: Theory Commitment Axioms.

Each philosophical theory (Physicalism, Functionalism, Dualism, IIT, GWT, HOT)
is captured as a set of Z3 axioms that define what that theory commits to.

These axioms are loaded into separate MicrotheoryManager contexts so
competing theories don't create logical explosions.

Reference: PHILOSOPHER_ENGINE_ARCHITECTURE.md, Layer 1
"""

from typing import Dict, List, Optional

try:
    from z3 import (
        Solver, Const, Consts, ForAll, Exists,
        Implies, And, Or, Not, BoolSort,
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

from .core import OntologySorts, OntologyRelations


THEORY_NAMES = [
    "physicalism",
    "property_dualism",
    "substance_dualism",
    "functionalism",
    "IIT",
    "GWT",
    "HOT",
]


class TheoryCommitments:
    """Axiomatized positions for each consciousness theory.

    Each method returns a list of Z3 constraints that define
    what a given theory asserts. These are loaded into separate
    solver contexts via MicrotheoryManager.
    """

    def __init__(self, sorts: OntologySorts, relations: OntologyRelations):
        if not Z3_AVAILABLE:
            raise ImportError("z3-solver required")
        self.S = sorts
        self.R = relations

    def physicalism(self) -> list:
        """Type-A Physicalism: mental = physical (identity).

        Core commitment: every mental property is identical to
        some physical property in every possible world.
        """
        S, R = self.S, self.R
        m = Const('m', S.Property)
        p = Const('p', S.Property)
        w = Const('w', S.World)

        return [
            # Supervenience: mental supervenes on physical in all worlds
            ForAll([m, w], Exists([p], R.Supervenes(m, p, w))),
            # Identity: mental IS physical (not just correlated)
            ForAll([m, w], Exists([p], R.IsIdentical(m, p, w))),
            # Causal closure: every physical event has a physical cause
            ForAll([Const('s1', S.State), Const('s2', S.State), w],
                   Implies(
                       R.CausesState(Const('s1', S.State),
                                     Const('s2', S.State), w),
                       True  # Physical cause exists (simplified)
                   )),
        ]

    def property_dualism(self) -> list:
        """Property Dualism: mental properties are non-physical but
        supervene on physical properties.

        Core commitment: supervenience without identity.
        Mental properties are real, irreducible, but depend on
        the physical.
        """
        S, R = self.S, self.R
        m = Const('m_pd', S.Property)
        p = Const('p_pd', S.Property)
        w = Const('w_pd', S.World)

        return [
            # Supervenience holds
            ForAll([m, w], Exists([p], R.Supervenes(m, p, w))),
            # But identity does NOT hold — mental ≠ physical
            Exists([m, w], Not(Exists([p], R.IsIdentical(m, p, w)))),
        ]

    def substance_dualism(self) -> list:
        """Cartesian Substance Dualism: mind and body are distinct
        substances with different principal attributes.

        This is Descartes' position: res cogitans (thinking substance)
        and res extensa (extended substance) are really distinct.
        """
        S, R = self.S, self.R

        mind_substance = Const('res_cogitans', S.Substance)
        body_substance = Const('res_extensa', S.Substance)
        thought = Const('thought', S.Property)
        extension = Const('extension', S.Property)

        return [
            # Two distinct substances
            mind_substance != body_substance,
            # Each has a principal attribute
            R.PrincipalAttribute(mind_substance, thought),
            R.PrincipalAttribute(body_substance, extension),
            # Principal attributes are distinct
            thought != extension,
            # Mind can exist without body (conceivability → possibility)
            Exists([Const('w_mind', S.World)], And(
                R.IsSubstance(S.ego, mind_substance,
                              Const('w_mind', S.World)),
                Not(R.HasProperty(S.ego, extension,
                                  Const('w_mind', S.World)))
            )),
        ]

    def functionalism(self) -> list:
        """Functionalism: mental states are functional states.

        Core commitment: what matters is the causal/functional role,
        not the physical substrate. A mental state is defined by
        its input-output profile.
        """
        S, R = self.S, self.R
        s1 = Const('s1_fn', S.State)
        s2 = Const('s2_fn', S.State)
        w = Const('w_fn', S.World)

        return [
            # Realization: mental states realized by physical states
            ForAll([s1, w], Exists([s2], R.Realizes(s2, s1, w))),
            # Multiple realizability: same mental, different physical
            Exists([s1, Const('w1', S.World), Const('w2', S.World)],
                   And(
                       R.Realizes(Const('phys1', S.State), s1,
                                  Const('w1', S.World)),
                       R.Realizes(Const('phys2', S.State), s1,
                                  Const('w2', S.World)),
                       Const('phys1', S.State) != Const('phys2', S.State),
                   )),
        ]

    def iit(self) -> list:
        """Integrated Information Theory (IIT / Tononi).

        Core commitment: consciousness = integrated information (phi).
        A system is conscious iff it has phi > 0.
        """
        S, R = self.S, self.R
        proc = Const('proc_iit', S.Process)
        exp = Const('exp_iit', S.Experience)
        w = Const('w_iit', S.World)

        return [
            # Consciousness requires integration
            ForAll([exp, w], Exists([proc], R.Integrates(proc, exp, w))),
            # Integration is gradable (phi > 0)
            # (simplified: existence of integration process)
        ]

    def gwt(self) -> list:
        """Global Workspace Theory (Baars/Dehaene).

        Core commitment: consciousness = global broadcast.
        Information becomes conscious when broadcast to
        the global workspace, making it available to
        all cognitive processes.
        """
        S, R = self.S, self.R
        s = Const('s_gwt', S.State)
        proc = Const('proc_gwt', S.Process)
        w = Const('w_gwt', S.World)

        return [
            # Consciousness requires broadcast
            ForAll([s, w],
                   Implies(
                       R.HasProperty(S.ego, Const('conscious', S.Property), w),
                       Exists([proc], R.BroadcastsTo(s, proc, w))
                   )),
        ]

    def hot(self) -> list:
        """Higher-Order Thought theory (Rosenthal).

        Core commitment: a mental state is conscious iff there
        is a higher-order thought about it.
        """
        S, R = self.S, self.R
        s1 = Const('s1_hot', S.State)
        s2 = Const('s2_hot', S.State)
        w = Const('w_hot', S.World)

        return [
            # Consciousness requires higher-order representation
            ForAll([s1, w],
                   Implies(
                       R.HasProperty(S.ego, Const('conscious', S.Property), w),
                       Exists([s2], R.HigherOrderOf(s2, s1, w))
                   )),
        ]

    def get_axioms(self, theory_name: str) -> list:
        """Get axioms for a named theory."""
        methods = {
            "physicalism": self.physicalism,
            "property_dualism": self.property_dualism,
            "substance_dualism": self.substance_dualism,
            "functionalism": self.functionalism,
            "IIT": self.iit,
            "GWT": self.gwt,
            "HOT": self.hot,
        }
        method = methods.get(theory_name)
        if method is None:
            raise ValueError(
                f"Unknown theory: {theory_name}. "
                f"Available: {list(methods.keys())}"
            )
        return method()
