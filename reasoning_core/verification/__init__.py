"""Layer 3: Verification Engine — Z3 multi-logic, CVC5 parallel, examples."""
from .z3_engine import (
    ModalLogicEngine, ParaconsistentEngine,
    DefeasibleEngine, MicrotheoryManager,
)
from .cvc5_engine import verify_parallel
