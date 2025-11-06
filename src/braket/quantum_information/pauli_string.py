# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from __future__ import annotations

import itertools
from cmath import exp, polar
from math import pi

from braket.circuits.circuit import Circuit
from braket.circuits.observables import I, Sum, TensorProduct, X, Y, Z

import numpy as np

_IDENTITY = "I"
_PAULI_X = "X"
_PAULI_Y = "Y"
_PAULI_Z = "Z"
_PAULI_INDICES = {_IDENTITY: 0, _PAULI_X: 1, _PAULI_Y: 2, _PAULI_Z: 3}
_PRODUCT_MAP = {
    "X": {"Y": ["Z", +1], "Z": ["Y", -1]},
    "Y": {"X": ["Z", -1], "Z": ["X", +1]},
    "Z": {"X": ["Y", +1], "Y": ["X", -1]},
}
_ID_OBS = I()
_PAULI_OBSERVABLES = {_PAULI_X: X(), _PAULI_Y: Y(), _PAULI_Z: Z()}
_SIGN_MAP = {"+": 1, "-": -1}
_EPS = 1e-10

numeric_types = (int, np.signedinteger, float, complex, np.floating, np.complexfloating)

class PauliString:  # noqa: PLR0904
    """A lightweight sparse representation of a Pauli string with its phase."""

    def __init__(self,
                 pauli_string: str | PauliString | tuple[int, dict[int, str]],
                 coeff: complex | tuple[float, float] = 1
                 ):
        """Initializes a length-n PauliString.

        Args:
            pauli_string (str | PauliString | tuple[int, dict[int, str]]):
                The representation of the pauli word, either a string (of the form)
                another PauliString object, or a tuple of (qubit_count, nontrivial_dict).
                A valid string consists of an (optional) coefficient with an uppercase string in
                {I, X, Y, Z}. Example valid strings are: XYZ, -YX, 0.5ZII, 0.3jIXY, 1e-3XX, etc.
            coeff (complex | tuple[float, float]): An optional coefficient that can be specified
                with the PauliString, can be either a coefficient or a tuple of polar coordinates
                (starting at 0 radians).

        Raises:
            ValueError: If the Pauli String is empty.
        """
        match coeff:
            case int() | np.signedinteger():
                self._modulus, self._phase = coeff, 0
            case float() | complex() | np.floating() | np.complexfloating():
                self._modulus, self._phase = polar(coeff)
            case tuple():
                self._modulus, self._phase = coeff
            case _:
                raise TypeError(
                    f" {coeff} must be of numeric type, not {type(coeff)}")
        match pauli_string:
            case PauliString():
                self._modulus *= pauli_string._modulus
                self._phase += pauli_string._phase
                self._qubit_count = pauli_string._qubit_count
                self._nontrivial = pauli_string._nontrivial
            case str():
                modulus, phase, factors_str = PauliString._split_pauli_string(pauli_string)
                self._phase += phase
                self._modulus *= modulus
                self._qubit_count = len(factors_str)
                self._nontrivial = PauliString._get_nontrivial(factors_str)
            case (int(qubit_count), dict(nontrivial)):
                self._qubit_count = qubit_count
                self._nontrivial = nontrivial
            case _:
                raise TypeError(
                    f"Pauli word {pauli_string} must be of type {PauliString}, {str}, or tuple")

    @property
    def phase(self) ->  float:
        """float: The complex phase of the PauliString. """
        return self._phase

    @property
    def coeff(self) -> complex | float:
        """ coefficient of the PauliString """
        return self._modulus * exp(1j * self._phase)

    @property
    def modulus(self) -> float:
        """ real norm of the PauliString """
        return self._modulus

    @property
    def qubit_count(self) -> int:
        """int: The number of qubits this Pauli string acts on."""
        return self._qubit_count

    @staticmethod
    def from_observable(
            observable: TensorProduct | I | X | Y | Z,
            nq: int | None = None) -> PauliString:
        """ Convert a tensor or single Pauli to a PauliString """

        has_targets = len(observable.targets) > 0
        if nq is None:
            nq = (max(observable.targets) + 1 if has_targets 
                  else getattr(observable, "__len__", [0].__len__)())
        if nq == 1:
            return PauliString(observable.ascii_symbols[0])
        pstr = {}
        if hasattr(observable, "factors"):
            for i, term in enumerate(observable.factors):
                qubit = int(term.targets) if has_targets else i
                pstr[qubit] = term.ascii_symbols[0]
        else:
            qubit = int(observable.targets[0])
            pstr[qubit] = observable.ascii_symbols[0]
        return PauliString((nq, pstr), observable.coefficient)

    def to_observable(self,
            include_trivial: bool = False) -> TensorProduct | I | X | Y | Z:
        """ Returns observable form of the Pauli string """
        if abs(self._phase * self._modulus - 1) < _EPS:
            return self.to_unsigned_observable(include_trivial=include_trivial)
        return self.coeff * self.to_unsigned_observable(
            include_trivial=include_trivial)

    def to_unsigned_observable(self,
            include_trivial: bool = False) -> TensorProduct | I | X | Y | Z:
        """Returns the observable corresponding to the unsigned part of the Pauli string.

        For example, for a Pauli string -XYZ, the corresponding observable is X ⊗ Y ⊗ Z.

        Args:
            include_trivial (bool): Whether to include explicit identity factors in the observable.
                Default: False.

        Returns:
            TensorProduct: The tensor product of the unsigned factors in the Pauli string.
        """
        if include_trivial:
            return TensorProduct([
                (
                    _PAULI_OBSERVABLES[self._nontrivial[qubit]]
                    if qubit in self._nontrivial
                    else _ID_OBS
                )
                for qubit in range(self._qubit_count)
            ])
        return TensorProduct([
            _PAULI_OBSERVABLES[self._nontrivial[qubit]] for qubit in sorted(self._nontrivial)
        ])

    def dot(self, other: PauliString | float, inplace: bool = False) -> PauliString:  # noqa: C901
        """Right multiplies this Pauli string with the argument.

        Returns the result of multiplying the current circuit by the argument on its right. For
        example, if called on `-XYZ` with argument `ZYX`, then `YIY` is the result. In-place
        computation is off by default.

        Args:
            other (PauliString): The right multiplicand.
            inplace (bool): If `True`, `self` is updated to hold the product.

        Returns:
            PauliString: The resultant circuit from right multiplying `self` with `other`.

        Raises:
            ValueError: If the lengths of the Pauli strings being multiplied differ.
        """
        if isinstance(other, numeric_types):
            modulus, phase = polar(other)
            if inplace:
                self._phase += phase
                self._modulus *= modulus
                return self
            return PauliString(
                (self._qubit_count, self._nontrivial.copy()),
                (modulus * self.modulus, phase + self.phase)
                )

        if self._qubit_count != other._qubit_count:
            raise ValueError(
                f"Input Pauli string must be of length ({self._qubit_count}), "
                f"not {other._qubit_count}"
            )

        nontrivial_result = {}
        phase_result = self._phase + other._phase
        moduli = self._modulus * other._modulus

        if not self.is_same_string(other):
            for i in self._nontrivial.keys() | other._nontrivial.keys():
                match (i in self._nontrivial, i in other._nontrivial):
                    case (False, True):
                        nontrivial_result[i] = other._nontrivial[i]
                    case (True, False):
                        nontrivial_result[i] = self._nontrivial[i]
                    case (True, True) if self._nontrivial[i] != other._nontrivial[i]:
                        gate, phase = _PRODUCT_MAP[self._nontrivial[i]][other._nontrivial[i]]
                        if gate != "I":
                            nontrivial_result[i] = gate
                        phase_result += phase * pi / 2

        if inplace:
            self._phase = phase_result % (2 * pi)
            self._modulus = moduli
            self._nontrivial = nontrivial_result
            return self
        return PauliString((self._qubit_count, nontrivial_result), (moduli, phase_result))

    def __mul__(self, other: PauliString | float) -> PauliString:
        """Right multiplication operator overload using `dot()`.

        Returns the result of multiplying the current circuit by the argument on its right.

        Args:
            other (PauliString): The right multiplicand.

        Returns:
            PauliString: The resultant circuit from right multiplying `self` with `other`.

        Raises:
            ValueError: If the lengths of the Pauli strings being multiplied differ.

        See Also:
            `braket.quantum_information.PauliString.dot()`
        """
        return self.dot(other)

    def __rmul__(self, other: float) -> PauliString:
        """Left multiplication operator overload for scalar multiplication.

        Handles cases like `2 * pauli_string`.

        Args:
            other (float): The scalar coefficient.

        Returns:
            PauliString: The resultant PauliString with updated coefficient.
        """
        return self.dot(other)

    def __imul__(self, other: PauliString | float) -> PauliString:
        """Operator overload for right-multiplication assignment (`*=`) using `dot()`.

        Right-multiplies `self` by `other`, and assigns the result to `self`.

        Args:
            other (PauliString): The right multiplicand.

        Returns:
            PauliString: The resultant circuit from right multiplying `self` with `other`.

        Raises:
            ValueError: If the lengths of the Pauli strings being multiplied differ.

        See Also:
            `braket.quantum_information.PauliString.dot()`
        """
        return self.dot(other, inplace=True)

    def power(self, n: int, inplace: bool = False) -> PauliString:
        """Composes Pauli string with itself n times.

        Args:
            n (int): Integer power of product.
            inplace (bool): Update `self` if `True`

        Returns:
            PauliString: If `n` is positive, result from self-multiplication `n` times.
            If zero, identity. If negative, self-multiplication from trivial
            inverse (recall Pauli operators are involutory).

        Raises:
            ValueError: If `n` isn't a plain Python `int`.
        """
        if not isinstance(n, int):
            raise TypeError("Must be raised to integer power")

        if inplace:
            self._phase *= n
            self._modulus **= n
            if n % 2 == 0:
                self._nontrivial = {}
            return self
        return PauliString(
            (self._qubit_count, {} if n % 2 == 0 else self._nontrivial.copy()),
            (self._modulus ** n, self._phase * n)
            )

    def __pow__(self, n: int) -> PauliString:
        """Pow operator overload for Pauli string composition.

        Syntactic sugar for `power()`.

        Args:
            n (int): The number of times to self-multiply. Can be any integer

        Returns:
            PauliString: If `n` is positive, result from self-multiplication `n` times.
            If zero, identity. If negative, self-multiplication from trivial
            inverse (recall Pauli operators are involutory).

        Raises:
            ValueError: If `n` isn't a plain Python `int`.

        See Also:
            `braket.quantum_information.PauliString.power()`
        """
        return self.power(n)

    def __ipow__(self, n: int) -> PauliString:
        """Operator overload for in-place pow assignment (`**=`) using `power()`.

        Syntactic sugar for in-place `power()`.

        Args:
            n (int): The number of times to self-multiply. Can be any integer

        Returns:
            PauliString: If `n` is positive, result from self-multiplication `n` times.
            If zero, identity. If negative, self-multiplication from trivial
            inverse (recall Pauli operators are involutory).

        Raises:
            ValueError: If `n` isn't a plain Python `int`.

        See Also:
            `braket.quantum_information.PauliString.power()`
        """
        return self.power(n, inplace=True)

    def weight_n_substrings(self, weight: int) -> tuple[PauliString, ...]:
        r"""Returns every substring of this Pauli string with exactly `weight` nontrivial factors.

        The number of substrings is equal to :math:`\binom{n}{w}`, where :math`n` is the number of
        nontrivial (non-identity) factors in the Pauli string and :math`w` is `weight`.

        Args:
            weight (int): The number of non-identity factors in the substrings.

        Returns:
            tuple[PauliString, ...]: A tuple of weight-n Pauli substrings.
        """
        substrings = []
        for indices in itertools.combinations(self._nontrivial, weight):
            factors = [
                (
                    self._nontrivial[qubit]
                    if qubit in set(indices).intersection(self._nontrivial)
                    else "I"
                )
                for qubit in range(self._qubit_count)
            ]
            substrings.append(
                PauliString(f"{''.join(factors)}")
            )
        return tuple(substrings)

    def eigenstate(self, signs: str | list[int] | tuple[int, ...] | None = None) -> Circuit:
        """Returns the eigenstate of this Pauli string with the given factor signs.

        The resulting eigenstate has each qubit in the +1 eigenstate of its corresponding signed
        Pauli operator. For example, a Pauli string +XYZ and signs ++- has factors +X, +Y and -Z,
        with the corresponding qubits in states `|+⟩` , `|i⟩` , and `|1⟩` respectively (the global
        phase of the Pauli string is ignored).

        Args:
            signs (str | list[int] | tuple[int, ...] | None): The sign of each factor of the
                eigenstate, specified either as a string of "+" and "_", or as a list or tuple of
                +/-1. The length of signs must be equal to the length of the Pauli string. If not
                specified, it is assumed to be all +. Default: None.

        Returns:
            Circuit: A circuit that prepares the desired eigenstate of the Pauli string.

        Raises:
            ValueError: If the length of signs is not equal to that of the Pauli string or the signs
                are invalid.
        """
        qubit_count = self._qubit_count
        if not signs:
            signs = "+" * qubit_count
        elif len(signs) != qubit_count:
            raise ValueError(
                f"signs must be the same length of the Pauli string ({qubit_count}), "
                f"but was {len(signs)}"
            )
        signs_tup = (
            tuple(_SIGN_MAP.get(sign) for sign in signs) if isinstance(signs, str) else tuple(signs)
        )
        if not set(signs_tup) <= {1, -1}:
            raise ValueError(f"signs must be +/-1, got {signs}")
        return self._generate_eigenstate_circuit(signs_tup)

    def to_circuit(self) -> Circuit:
        """Returns circuit represented by this `PauliString`.

        Returns:
            Circuit: The circuit for this `PauliString`.
        """
        assert abs(self._modulus - 1) <= _EPS, "Only unit coefficients are supported"  # noqa: S101
        assert self._phase in {0, pi}, "Only unit phases are supported"  # noqa: S101

        circ = Circuit()
        for qubit in range(self._qubit_count):
            match self._nontrivial.get(qubit, "I"):
                case "I":
                    circ.i(qubit)
                case "X":
                    circ.x(qubit)
                case "Y":
                    circ.y(qubit)
                case "Z":
                    circ.z(qubit)
        # if self._phase != 1:
        #     circ.global_phase(self._phase)
        return circ

    def __eq__(self, other: PauliString):
        if isinstance(other, PauliString):
            return (
                abs(self.coeff - other.coeff) <= _EPS
                and self._nontrivial == other._nontrivial
                and self._qubit_count == other._qubit_count
            )
        return False

    def is_same_string(self, other: PauliString) -> bool:
        if isinstance(other, PauliString):
            return (self._nontrivial == other._nontrivial
                and self._qubit_count == other._qubit_count
            )
        return False

    def __getitem__(self, item: int):
        if item >= self._qubit_count:
            raise IndexError(item)
        return _PAULI_INDICES[self._nontrivial.get(item, "I")]

    def __len__(self):
        """ total length, i.e. number of qubits """
        return self._qubit_count

    @property
    def degree(self) -> int:
        """ number of non trivial Pauli elements in the string """
        return len(self._nontrivial)

    def __str__(self) -> str:
        """ shorter output form without coefficient """
        return "".join([self._nontrivial.get(qubit, "I") for qubit in range(self._qubit_count)])

    def __repr__(self):
        """ can be sued to self replicate """
        factors = [self._nontrivial.get(qubit, "I") for qubit in range(self._qubit_count)]
        return f"{self.coeff}{''.join(factors)}"

    @staticmethod
    def _get_nontrivial(pauli_str: str) -> dict[int, str]:
        return {i: p for i, p in enumerate(pauli_str) if pauli_str[i] != "I"}

    def _generate_eigenstate_circuit(self, signs: tuple[int, ...]) -> Circuit:
        circ = Circuit()
        for qubit in range(len(signs)):
            state = signs[qubit] * self[qubit]
            if state == -3:
                circ.x(qubit)
            elif state == 1:
                circ.h(qubit)
            elif state == -1:
                circ.x(qubit).h(qubit)
            elif state == 2:
                circ.h(qubit).s(qubit)
            elif state == -2:
                circ.h(qubit).si(qubit)
            # circ.global_phase(self.phase)
        return circ

    @staticmethod
    def _split_pauli_string(pauli_word: str) -> tuple[float, float, str]:
        """ split a string into a coefficient or sign and Pauli expression """
        for i, char in enumerate(pauli_word):
            if char in _PAULI_INDICES:
                coeff_str = pauli_word[:i]
                pauli_str = pauli_word[i:]
                match coeff_str:
                    case "" | "+":
                        modulus, phase = 1, 0
                    case "-":
                        modulus, phase = 1, pi
                    case _:
                        modulus, phase = polar(complex(coeff_str))
                break
        else:
            raise ValueError(f"{pauli_word} is not a valid Pauli string")

        if not pauli_str:
            raise ValueError("Pauli string cannot be empty")
        if set(pauli_str) - _PAULI_INDICES.keys():
            raise ValueError(f"{pauli_word} is not a valid Pauli string")
        return modulus, phase, pauli_str
