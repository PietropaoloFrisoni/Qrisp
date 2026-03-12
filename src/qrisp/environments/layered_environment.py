"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

from typing import List, Tuple

import jax.core as jc
from jax.extend.core import ClosedJaxpr, JaxprEqn, Literal, Var

from qrisp.circuit.instruction import Instruction
from qrisp.environments.quantum_environments import QuantumEnvironment
from qrisp.jasp.interpreter_tools.abstract_interpreter import ContextDict
from qrisp.jasp.jasp_expression import Jaspr
from qrisp.jasp.primitives import AbstractQuantumState


class GateStack(QuantumEnvironment):
    """
    Collect operations into layers for a parent :class:`LayeredEnvironment` to interleave.

    This class intentionally does not emit instructions to the parent environment when they are added.
    Instead, it collects them into layers, which are
    emitted by the parent :class:`LayeredEnvironment` after interleaving.

    For more details and examples, see :class:`LayeredEnvironment`.

    """

    def __init__(self, env_args=None):
        """Initialize with empty layers."""

        super().__init__(env_args=env_args)

        self.layers: List[Instruction] = []

    def compile(self):
        """
        Prepare layers, but do not emit anything.

        The parent LayeredEnvironment will interleave and emit these layers.
        """

        if self.parent is None:
            raise ValueError(
                "GateStack must have a parent LayeredEnvironment to compile"
            )

        for instr in self.env_data:
            if isinstance(instr, QuantumEnvironment):
                raise ValueError("Nested QuantumEnvironments are not supported")

            self.layers.append(instr)


class LayeredEnvironment(QuantumEnvironment):
    """
    A QuantumEnvironment that interleaves layers from consecutive ``GateStack``
    children to reduce circuit depth.

    Instructions inside each ``GateStack`` are treated as ordered layers.
    When two or more ``GateStack`` objects appear consecutively in the instruction
    stream, their layers are emitted in a "brick" pattern: all layer-0 instructions
    across every stack first, then all layer-1 instructions, and so on.  Because
    operations from different stacks are reordered, the user is responsible for
    ensuring that instructions across adjacent stacks act on disjoint qubits or
    otherwise commute.

    Any instruction emitted directly inside ``LayeredEnvironment`` and outside of a
    ``GateStack`` is passed through immediately at the position it appears,
    breaking adjacency between the stacks on either side of it. Stacks separated
    by such a bare instruction are therefore interleaved independently, not with
    each other.

    If two stacks have different numbers of layers, the shorter stack simply
    contributes nothing for the extra layers of the longer one.

    .. note::

        This environment performs no commutation or qubit-conflict checks.
        Incorrectly interleaving non-commuting operations on shared qubits will
        produce a wrong circuit.

    Parameters
    ----------
    env_args : optional
        Forwarded to the parent :class:`QuantumEnvironment` constructor.

    Examples
    --------

    **Basic usage: parallel stabilizer measurement**

    Consider measuring a Z-type stabilizer by entangling data qubits with ancilla
    qubits via CX gates.  We define interleaved data and ancilla qubits

    ::

        from qrisp import *

        n = 3
        qubits = QuantumArray(qtype=QuantumBool(), shape=(2 * n - 1))
        data_qubits   = qubits[::2]   # even indices: data
        ancilla_qubits = qubits[1::2]  # odd  indices: ancilla

    Without layering, instructions are emitted in source order. That is, all CX gates for
    each stabilizer are grouped together, giving circuit depth :math:`2(n-1)`:

    ::

        for i in range(n - 1):
            cx(data_qubits[i],     ancilla_qubits[i])
            cx(data_qubits[i + 1], ancilla_qubits[i])

    >>> print(qubits.qs)
        QuantumCircuit:
        ---------------
          qubits.0: ──■─────────────────
                    ┌─┴─┐┌───┐
        qubits_1.0: ┤ X ├┤ X ├──────────
                    └───┘└─┬─┘
        qubits_2.0: ───────■────■───────
                              ┌─┴─┐┌───┐
        qubits_3.0: ──────────┤ X ├┤ X ├
                              └───┘└─┬─┘
        qubits_4.0: ─────────────────■──
        ...

    >>> print(qubits.qs.depth())
    4

    Wrapping each stabilizer in a ``GateStack`` inside a ``LayeredEnvironment``
    interleaves the CX gates across stabilizers, reducing depth to 2:

    ::

        qubits.qs.clear_data()

        with LayeredEnvironment():
            for i in range(n - 1):
                with GateStack():
                    cx(data_qubits[i],     ancilla_qubits[i])
                    cx(data_qubits[i + 1], ancilla_qubits[i])

    >>> print(qubits.qs)
    QuantumCircuit:
    ---------------
      qubits.0: ──■───────
                ┌─┴─┐┌───┐
    qubits_1.0: ┤ X ├┤ X ├
                └───┘└─┬─┘
    qubits_2.0: ──■────■──
                ┌─┴─┐┌───┐
    qubits_3.0: ┤ X ├┤ X ├
                └───┘└─┬─┘
    qubits_4.0: ───────■──
    ...

    >>> print(qubits.qs.depth())
    2

    **Bare instructions break adjacency**

    A bare instruction between two ``GateStack`` objects is emitted at that exact
    position in the output, and it breaks the run of adjacent stacks. Stacks on
    either side of it are not interleaved with each other:

    ::

        N = 4
        qubits = QuantumArray(qtype=QuantumBool(), shape=(2 * N - 1))
        data_qubits    = qubits[::2]
        ancilla_qubits = qubits[1::2]

        with LayeredEnvironment():
            with GateStack():
                cx(data_qubits[0], ancilla_qubits[0])
                cx(data_qubits[1], ancilla_qubits[0])
            with GateStack():
                cx(data_qubits[1], ancilla_qubits[1])
                cx(data_qubits[2], ancilla_qubits[1])

            z(data_qubits[2])

            with GateStack():
                cx(data_qubits[2], ancilla_qubits[2])
                cx(data_qubits[3], ancilla_qubits[2])

    The first two stacks are interleaved with each other. Then, the Z gate is emitted
    immediately after. Finally, the last stack is emitted sequentially on its own:

    >>> print(qubits.qs)
    QuantumCircuit:
    ---------------
      qubits.0: ──■──────────────────────
                ┌─┴─┐┌───┐
    qubits_1.0: ┤ X ├┤ X ├───────────────
                └───┘└─┬─┘
    qubits_2.0: ──■────■─────────────────
                ┌─┴─┐┌───┐
    qubits_3.0: ┤ X ├┤ X ├───────────────
                └───┘└─┬─┘┌───┐
    qubits_4.0: ───────■──┤ Z ├──■───────
                          └───┘┌─┴─┐┌───┐
    qubits_5.0: ───────────────┤ X ├┤ X ├
                               └───┘└─┬─┘
    qubits_6.0: ──────────────────────■──
    ...

    """

    def __init__(self, env_args=None) -> None:
        super().__init__(env_args=env_args)

    def _emit_instruction(self, instr: Instruction) -> None:
        """Emit a single instruction into env_qs."""

        if isinstance(instr, QuantumEnvironment):
            raise ValueError("Nested QuantumEnvironments are not supported")

        self.env_qs.append(instr)

    # We interleave by grouping layers by their index across all stacks.
    #
    # For example, if we have:
    #
    # Stack A: layers [A1, A2, A3]
    # Stack B: layers [B1]
    # Stack C: layers [C1, C2]
    #
    # The interleaved order would be: A1, B1, C1, A2, C2, A3
    def _interleave_layers(self, stacks: List[GateStack]) -> None:
        """Interleave layers from multiple GateStacks in brick order."""

        if not stacks:
            return

        max_layers = max((len(st.layers) for st in stacks), default=0)

        for layer_idx in range(max_layers):
            for st in stacks:
                if layer_idx < len(st.layers):
                    instr = st.layers[layer_idx]
                    self._emit_instruction(instr)

    # Each segment is a tuple of the form (segment_type, items), where segment_type is either "bare" or "stacks":
    #
    # - "bare": items is a list of instructions that are not GateStacks.
    # - "stacks": items is a list of consecutive GateStack objects.
    #
    # For example, if self.env_data contains the following sequence of instructions:
    #
    # [instr1, instr2, GateStack1, GateStack2, instr3, GateStack3]
    #
    # The resulting segments would be:
    #
    # [
    #     ("bare", [instr1, instr2]),
    #     ("stacks", [GateStack1, GateStack2]),
    #     ("bare", [instr3]),
    #     ("stacks", [GateStack3])
    # ]
    def _prepare_segment(self) -> List[Tuple[str, List]]:
        """
        Prepare a list of segments from self.env_data, grouping consecutive GateStacks together.
        """

        segments: List[Tuple[str, List]] = []

        for instr in self.env_data:
            is_gatestack = isinstance(instr, GateStack)
            segment_type = "stacks" if is_gatestack else "bare"

            if segments and segments[-1][0] == segment_type:
                segments[-1][1].append(instr)
            else:
                segments.append((segment_type, [instr]))

        return segments

    def compile(self) -> None:
        """Compile by interleaving GateStack layers in brick order."""

        segments = self._prepare_segment()

        for segment_type, items in segments:
            if segment_type == "bare":
                for instr in items:
                    self._emit_instruction(instr)
            else:
                for stack in items:
                    stack.compile()
                self._interleave_layers(items)

    def jcompile(self, eqn: JaxprEqn, context_dic: ContextDict) -> None:
        """jcompile by flattening the inner jaspr of the q_env/GateStack and evaluating it with the given context_dic."""

        from qrisp.jasp import eval_jaxpr, extract_invalues, insert_outvalues

        flat_jaspr = flatten_layered_environment(eqn)
        print("Flattened Jaspr for LayeredEnvironment:")
        print(flat_jaspr)
        args = extract_invalues(eqn, context_dic)
        res = eval_jaxpr(flat_jaspr)(*args)
        res = (res,) if not isinstance(res, tuple) else res
        insert_outvalues(eqn, context_dic, res)


def _is_quantum_state(var):
    """Check if a variable holds a QuantumState."""
    return isinstance(var.aval, AbstractQuantumState)


def _substitute_eqn(eqn, subst):
    new_invars = []
    for v in eqn.invars:
        if isinstance(v, Literal):
            new_invars.append(v)
        else:
            new_invars.append(subst.get(v, v))

    new_outvars = []
    for v in eqn.outvars:
        if isinstance(v, jc.DropVar) and not _is_quantum_state(v):
            # Keep DropVar only for non-state outputs
            new_outvars.append(v)
        else:
            # Always generate a fresh var for QuantumState outputs,
            # even if the original was a DropVar
            fresh = Var(v.aval)
            subst[v] = fresh
            new_outvars.append(fresh)

    return JaxprEqn(
        invars=new_invars,
        outvars=new_outvars,
        primitive=eqn.primitive,
        params=eqn.params,
        effects=eqn.effects,
        source_info=eqn.source_info,
        ctx=eqn.ctx,
    )


def _prepare_segments(eqns: List[JaxprEqn]) -> List[Tuple[str, List[JaxprEqn]]]:
    """
    Segment a flat list of equations into runs of GateStacks and bare instructions.

    Returns
    -------
    list of ("bare" | "stacks", list of JaxprEqn)
    """
    segments = []
    for eqn in eqns:
        is_gatestack = (
            eqn.primitive.name == "jasp.q_env" and eqn.params.get("type") == "GateStack"
        )
        seg_type = "stacks" if is_gatestack else "bare"

        if segments and segments[-1][0] == seg_type:
            segments[-1][1].append(eqn)
        else:
            segments.append((seg_type, [eqn]))

    return segments


def _interleave_gatestack_segment(
    stack_eqns: List[JaxprEqn], incoming_state_var: Var
) -> Tuple[List[JaxprEqn], Var]:
    """
    Interleave layers from consecutive GateStack equations in brick order,
    lifting all inner equations into the outer namespace.

    Parameters
    ----------
    stack_eqns : list of JaxprEqn
        Consecutive q_env/GateStack equations.
    incoming_state_var : Var
        The QuantumState entering this block.


    Returns
    -------
    list of JaxprEqn
        Interleaved, lifted equations.
    Var
        The final outgoing QuantumState variable.
    """
    # Extract raw inner eqns per stack (still in inner namespace)
    stacks_inner_eqns = [eqn.params["jaspr"].eqns for eqn in stack_eqns]
    max_layers = max(len(layers) for layers in stacks_inner_eqns)

    # Build interleaved sequence of (stack_idx, layer_eqn) pairs
    interleaved_pairs = []
    for layer_idx in range(max_layers):
        for stack_idx, layers in enumerate(stacks_inner_eqns):
            if layer_idx < len(layers):
                interleaved_pairs.append((stack_idx, layers[layer_idx]))

    # Build one substitution map per stack for non-state invars
    # The QuantumState invar is threaded dynamically below
    subst_maps = []
    for stack_eqn in stack_eqns:
        inner_jaspr = stack_eqn.params["jaspr"]
        inner_invars = list(inner_jaspr.invars)
        outer_invars = list(stack_eqn.invars)
        subst = {}
        for inner_v, outer_v in zip(inner_invars[:-1], outer_invars[:-1]):
            subst[inner_v] = outer_v
        subst_maps.append(subst)

    current_state = incoming_state_var
    lifted_eqns = []

    for stack_idx, inner_eqn in interleaved_pairs:
        subst = subst_maps[stack_idx]

        # Override the QuantumState input of THIS specific equation,
        # not just the jaspr-level state invar.
        # This correctly handles both the first equation (which consumes the
        # jaspr invar) and subsequent ones (which consume local state vars).
        for v in inner_eqn.invars:
            if (
                not isinstance(v, Literal)
                and not isinstance(v, jc.DropVar)
                and _is_quantum_state(v)
            ):
                subst[v] = current_state

        new_eqn = _substitute_eqn(inner_eqn, subst)
        lifted_eqns.append(new_eqn)

        for outvar in new_eqn.outvars:
            if not isinstance(outvar, jc.DropVar) and _is_quantum_state(outvar):
                current_state = outvar

    return lifted_eqns, current_state


def _rewire_bare_eqn(
    eqn: JaxprEqn, incoming_state_var: Var, Var: type
) -> Tuple[JaxprEqn, Var]:
    """
    Re-wire a single bare (non-GateStack) equation's QuantumState
    to use incoming_state_var, producing a fresh outgoing state variable.
    """
    new_invars = []
    for v in eqn.invars:
        if isinstance(v, Literal):
            new_invars.append(v)
        elif not isinstance(v, jc.DropVar) and _is_quantum_state(v):
            new_invars.append(incoming_state_var)
        else:
            new_invars.append(v)

    new_outvars = []
    outgoing_state = incoming_state_var
    for v in eqn.outvars:
        if isinstance(v, jc.DropVar):
            new_outvars.append(v)
        elif _is_quantum_state(v):
            fresh = Var(v.aval)
            new_outvars.append(fresh)
            outgoing_state = fresh
        else:
            new_outvars.append(v)

    return (
        JaxprEqn(
            invars=new_invars,
            outvars=new_outvars,
            primitive=eqn.primitive,
            params=eqn.params,
            effects=eqn.effects,
            source_info=eqn.source_info,
            ctx=eqn.ctx,
        ),
        outgoing_state,
    )


def flatten_layered_environment(q_env_eqn: JaxprEqn) -> Jaspr:
    """
    Takes the outer q_env equation for a LayeredEnvironment, extracts its
    inner jaspr, performs the layer interleaving, and returns a new flat
    jaspr ready for execution.
    """

    outer_jaspr = q_env_eqn.params["jaspr"]

    print(f"jaspr received by `flatten_layered_environment`: {outer_jaspr}")

    # The incoming QuantumState is always the last invar of the outer jaspr
    # by Jaspr convention — no need to look it up from context_dic
    current_state = outer_jaspr.invars[-1]

    segments = _prepare_segments(outer_jaspr.eqns)
    new_eqns = []

    for seg_type, items in segments:
        if seg_type == "bare":
            for eqn in items:
                new_eqn, current_state = _rewire_bare_eqn(eqn, current_state, Var)
                new_eqns.append(new_eqn)
        else:
            interleaved, current_state = _interleave_gatestack_segment(
                items, current_state
            )
            new_eqns.extend(interleaved)

    return Jaspr(
        constvars=list(outer_jaspr.constvars),
        invars=list(outer_jaspr.invars),
        outvars=list(outer_jaspr.outvars[:-1]) + [current_state],
        eqns=new_eqns,
        consts=list(outer_jaspr.consts),
        debug_info=outer_jaspr.debug_info,
    )
