"""
Layer 1 Core: OWL 2 Ontology Sorts and Relations.

Defines the formal vocabulary for philosophy of mind:
- Sorts: World, Property, Subject, State, Experience, Process
- Relations: HasProperty, CausesState, Supervenes, AccessibleFrom, etc.

Everything downstream — argumentation, verification, formalizations —
uses this vocabulary.

Reference: PHILOSOPHER_ENGINE_ARCHITECTURE.md, Layer 1
"""

from typing import Dict, Optional

try:
    from z3 import (
        DeclareSort, Function, BoolSort, IntSort,
        Const, Consts, ForAll, Implies, And, Or, Not,
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


class OntologySorts:
    """Z3 sorts for the philosophical domain.

    Six fundamental sorts capturing the ontological categories
    relevant to philosophy of mind and Cartesian metaphysics.
    """

    def __init__(self):
        if not Z3_AVAILABLE:
            raise ImportError("z3-solver required: pip install z3-solver")

        # Core sorts
        self.World = DeclareSort('World')
        self.Property = DeclareSort('Property')
        self.Subject = DeclareSort('Subject')
        self.State = DeclareSort('State')
        self.Experience = DeclareSort('Experience')
        self.Process = DeclareSort('Process')

        # Substance sorts (Cartesian)
        self.Substance = DeclareSort('Substance')
        self.Mode = DeclareSort('Mode')

        # Common constants
        self.actual_world = Const('actual', self.World)
        self.ego = Const('ego', self.Subject)

    def get_sort(self, name: str):
        """Get a sort by name."""
        return getattr(self, name, None)

    def all_sorts(self) -> Dict[str, 'z3.SortRef']:
        return {
            'World': self.World,
            'Property': self.Property,
            'Subject': self.Subject,
            'State': self.State,
            'Experience': self.Experience,
            'Process': self.Process,
            'Substance': self.Substance,
            'Mode': self.Mode,
        }


class OntologyRelations:
    """Z3 relations between ontological entities.

    Captures the structural relationships used across
    all philosophical theories in the system:
    - HasProperty: subject-property attribution
    - CausesState: causal relations
    - Supervenes: supervenience (physicalist claim)
    - AccessibleFrom: modal accessibility (possible worlds)
    - IsIdentical: identity relation
    - Realizes: functional realization
    - Integrates: IIT-style information integration
    """

    def __init__(self, sorts: OntologySorts):
        if not Z3_AVAILABLE:
            raise ImportError("z3-solver required: pip install z3-solver")

        self.sorts = sorts
        S = sorts

        # Core relations
        self.HasProperty = Function(
            'HasProperty', S.Subject, S.Property, S.World, BoolSort())

        self.CausesState = Function(
            'CausesState', S.State, S.State, S.World, BoolSort())

        self.Supervenes = Function(
            'Supervenes', S.Property, S.Property, S.World, BoolSort())

        self.AccessibleFrom = Function(
            'AccessibleFrom', S.World, S.World, BoolSort())

        self.IsIdentical = Function(
            'IsIdentical', S.Property, S.Property, S.World, BoolSort())

        # Theory-specific relations
        self.Realizes = Function(
            'Realizes', S.State, S.State, S.World, BoolSort())

        self.Integrates = Function(
            'Integrates', S.Process, S.Experience, S.World, BoolSort())

        self.HigherOrderOf = Function(
            'HigherOrderOf', S.State, S.State, S.World, BoolSort())

        self.BroadcastsTo = Function(
            'BroadcastsTo', S.State, S.Process, S.World, BoolSort())

        # Cartesian-specific
        self.IsSubstance = Function(
            'IsSubstance', S.Subject, S.Substance, S.World, BoolSort())

        self.HasMode = Function(
            'HasMode', S.Substance, S.Mode, S.World, BoolSort())

        self.PrincipalAttribute = Function(
            'PrincipalAttribute', S.Substance, S.Property, BoolSort())

        # Causal adequacy (Trademark argument)
        self.FormalReality = Function(
            'FormalReality', S.Subject, IntSort())

        self.ObjectiveReality = Function(
            'ObjectiveReality', S.Subject, IntSort())

        self.Causes = Function(
            'Causes', S.Subject, S.Subject, BoolSort())

    def get_relation(self, name: str):
        """Get a relation by name."""
        return getattr(self, name, None)

    def all_relations(self) -> Dict[str, 'z3.FuncDeclRef']:
        return {
            name: getattr(self, name)
            for name in [
                'HasProperty', 'CausesState', 'Supervenes',
                'AccessibleFrom', 'IsIdentical', 'Realizes',
                'Integrates', 'HigherOrderOf', 'BroadcastsTo',
                'IsSubstance', 'HasMode', 'PrincipalAttribute',
                'FormalReality', 'ObjectiveReality', 'Causes',
            ]
        }


class OWLTaxonomy:
    """OWL 2 classification layer with Zalta's dual predication.

    In Zalta's abstract object theory:
    - Ordinary objects EXEMPLIFY properties
    - Abstract objects ENCODE properties

    This distinction matters for:
    - Descartes' idea of God (encodes perfection, doesn't exemplify it)
    - Mathematical objects (encode properties without spatial location)
    - Fictional entities (Sherlock Holmes encodes detective-hood)
    """

    def __init__(self, sorts: OntologySorts):
        if not Z3_AVAILABLE:
            raise ImportError("z3-solver required")

        self.sorts = sorts
        S = sorts

        # Zalta dual predication
        self.Exemplifies = Function(
            'Exemplifies', S.Subject, S.Property, S.World, BoolSort())
        self.Encodes = Function(
            'Encodes', S.Subject, S.Property, BoolSort())

        # OWL 2 class membership
        self.IsA = Function(
            'IsA', S.Subject, DeclareSort('OWLClass'), BoolSort())

    def abstract_object_axiom(self):
        """Abstract objects can encode properties they don't exemplify.

        Key: the idea of God encodes infinite perfection
        but doesn't itself have infinite formal reality.
        """
        S = self.sorts
        x = Const('x', S.Subject)
        p = Const('p', S.Property)
        w = Const('w', S.World)

        # Encoding doesn't require exemplification in any world
        return ForAll(
            [x, p],
            Not(Implies(
                self.Encodes(x, p),
                self.Exemplifies(x, p, S.actual_world)
            ))
        )
